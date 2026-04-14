import cupy as cp
import numpy as np
from vispy import app, scene, io
import matplotlib.pyplot as plt # Para el mapa de colores
import imageio
import time

vertice_seleccionado = None  # Numero de vertice
start_x_mouse = None 
start_y_mouse = None
grupo_seleccionado = None
lado_seleccionado = None
indices_afectados = []

#-------------------------------------------------------
# CONFIGURACIONE DE VIDIO 
#-------------------------------------------------------
NOMBRE_VIDEO = 'simulacion_resortes.mp4'
DURACION_SEGUNDOS = 10
FPS = 30
TOTAL_FRAMES = DURACION_SEGUNDOS * FPS
MODO_VIDEO = True
SUB_STEPS = 20
TAMAÑO_PUNTOS = 0.1

distancia_vista = 15
ruta = "tambor.obj"   # 11991 vertices
limite_fuerza = 0.2
frame_espera = 0

#-------------------------------------------------------
# CONSTANTES RESORTES 
#-------------------------------------------------------
L0 = 0.0         # longitud natural del resorte     1
k = 50.0      # constante del resorte
m = 0.05      # masa de cada punto
dt = 0.005     # delta de tiempo
damping = 0.999 # perdida de energia
e = 0.9       # constante de rebote
g = 0

#-------------------------------------------------------
# CARGAR OBJ 
#-------------------------------------------------------
def cargar_obj(ruta):
    vertices = []
    caras_vuelta = []
    with open(ruta, 'r') as f:
        for linea in f:
            # Guardamos los vertices
            if linea.startswith('v '):                                  
                _, x, y, z = linea.strip().split()
                vertices.append([float(x), float(y), float(z)])
            # Guardamos los vertices que conforman las caras
            elif linea.startswith('f '):
                partes = linea.strip().split()[1:]
                indices = [int(p.split('/')[0]) - 1 for p in partes] #"f 1/1/1" p.split('/') → ['1', '1', '1']                                
                caras_vuelta.append(indices + [indices[0]])          # Es -1 para que coincida con el indice de aqui 
    return np.array(vertices, dtype=np.float32), caras_vuelta

#-------------------------------------------------------
# OBTENER VECINOS
#-------------------------------------------------------
def obtener_vecinos(caras, n_vertices):
    vecinos = [set() for _ in range(n_vertices)]

    for cara in caras:
        for i in range(len(cara) - 1):
            v1 = cara[i]
            v2 = cara[i+1]

            # Agregar ambos sentidos
            vecinos[v1].add(v2)
            vecinos[v2].add(v1)

    # Convertir a listas
    vecinos = [list(v) for v in vecinos]
    return vecinos


#-------------------------------------------------------
# CARGA INICIAL 
#-------------------------------------------------------
pos, caras_vuelta = cargar_obj(ruta)  # Pos para cada vértice: [[x,y,z]], Caras para cada cara: [[p1,p2,p3,p4]]
vecinos = obtener_vecinos(caras_vuelta, len(pos))

# Inicializamos los arreglos de velocidades y fuerzas en 0 [[0, 0, 0]]
pos_gpu = cp.asarray(pos, dtype=cp.float32)
vel_gpu = cp.zeros((len(pos), 3), dtype=cp.float32)
fue_gpu = cp.zeros((len(pos), 3), dtype=cp.float32)
# Array de los vertices fijos que no moveremos
fijos_gpu = cp.zeros(len(pos), dtype=cp.int32) # Todo empieza en 0 (libre)

# -------------------------------------------------------
# PRE-CALCULO DE INDICES DE ARISTAS 
# -------------------------------------------------------
indices_lineas = []
for i in range(len(caras_vuelta)):
    n_vertices = len(caras_vuelta[i])
    for j in range(n_vertices):
        if j < n_vertices - 1:
            v1 = caras_vuelta[i][j]
            v2 = caras_vuelta[i][j+1]
            # Guardamos SOLO LOS INDICES, no las posiciones
            indices_lineas.extend([v1, v2])

# Lo convertimos a numpy array para usarlo como "mapa" rapido
indices_lineas_np = np.array(indices_lineas, dtype=np.int32)

#-------------------------------------------------------
# VISPY 
#-------------------------------------------------------
# Crear la ventana/canvas
canvas = scene.SceneCanvas(keys='interactive', show=False, size=(1920, 1080),config=dict(samples=4), bgcolor='white')
# Crear una "vista" dentro del canvas         
view = canvas.central_widget.add_view()      
# Cámara para interactuar                       
view.camera = scene.cameras.TurntableCamera(fov=45, distance=distancia_vista)     
# Puntos
#scatter = scene.visuals.Markers(pos=pos, size=TAMAÑO_PUNTOS, face_color='red', parent=view.scene,symbol='disc',
                                #edge_width=0, spherical=False,light_color='white',scaling=True)
# Aristas
line = scene.visuals.Line(pos=pos[indices_lineas_np], color='white', width=4, connect='segments', parent=view.scene)
# Ejes
from vispy.scene.visuals import XYZAxis                              # dibujar ejes
from vispy.visuals.transforms import STTransform            
axis = XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(5, 5, 5), translate=(0, 0, 0))
view.camera.center = pos.mean(axis=0)                                # centar camara alrededor del objeto

# ... (configuracion de vispy anterior) ...
view.camera = scene.cameras.TurntableCamera(fov=45, distance=distancia_vista) 

# --- AGREGAR ESTO ---
view.camera.elevation = 0   # 0 grados = Vista totalmente horizontal (ni desde arriba ni desde abajo)
view.camera.azimuth = 90

#-------------------------------------------------------
# CALCULO DE FUERZAS Y POSICIONES CON PARALELISMO
#-------------------------------------------------------

new_fuerzas = cp.RawKernel(r'''
extern "C" __global__
void kernel_test(
    float3 *fuerza,
    float3 *pos,
    int *indice_start,
    int *vecinos_list,
    int num_vertices,
    float k,
    float L0
){
                                                            // cantidad_threads_por_bloque * idx_bloque_en_el_grid + idx_hilo_en_su_bloque
    int i = blockDim.x * blockIdx.x + threadIdx.x;          // indice global
    if (i >= num_vertices) return;
    
    float3 pos_yo = pos[i];                                 // mi posicion
    float3 F = make_float3(0,0,0);                          // hacer fuerzas 0 para calcular una nueva

    int start = indice_start[i];                            // en que indice empiezan mis vecinos
    int end   = indice_start[i+1];                          // en que indice terminan mis vecinos

    for (int t = start; t < end; t++){                      // sabiendo donde inician y terminan los vecinos
        int j = vecinos_list[t];                            // recorrer un for por cada uno en vecinos_list
        float3 pos_vecino = pos[j];                         // posicion de mi vecino

        float3 d = make_float3(pos_vecino.x - pos_yo.x,     // distancia entre mi VECINO y YO 
                               pos_vecino.y - pos_yo.y,
                               pos_vecino.z - pos_yo.z);

        float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);

        // Normalizar
        d.x /= dist;
        d.y /= dist;
        d.z /= dist;
        
        F.x += k * (dist - L0) * d.x;
        F.y += k * (dist - L0) * d.y;
        F.z += k * (dist - L0) * d.z;
    }

    fuerza[i] = F;
}
''', 'kernel_test')

new_pos = cp.RawKernel(r'''
extern "C" __global__
void kernel_test(
    float3 *fuerza,
    float3 *pos,
    float3 *vel,
    int *fijos,
    int num_vertices,
    float m,
    float g,
    float damping,
    float e,
    float dt
){
                                                     
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // cantidad_threads_por_bloque * idx_bloque_en_el_grid + idx_hilo_en_su_bloque
    if (i >= num_vertices) return;                   // indice global

    
    // --- LOGICA DE FIJADO ---
    // Si fijos[i] es 1, matamos la velocidad y nos salimos.
    if (fijos[i] == 1) {
        vel[i] = make_float3(0,0,0); 
        return; 
    }
    // ------------------------
    
    float3 F = fuerza[i];                            // mi fuerza
    float3 a = make_float3(F.x/m, F.y/m, F.z/m);     // aceleracion en x,y,z

    // velocidad actualizada con las fuerza
    vel[i].x += a.x * dt;                                 
    vel[i].y += a.y * dt;
    vel[i].z += a.z * dt;
    // amortiguacion
    vel[i].x *= damping;                                 
    vel[i].y *= damping;
    vel[i].z *= damping;
    // gravedad
    vel[i].z -= g * dt;

    // con velocidades actualizadas cambiar posicion
    pos[i].x += vel[i].x * dt;                              
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;
    
    // colision con el suelo z = 0
    if (pos[i].z < 0.05){ 
        pos[i].z = 0;
        vel[i].z = -vel[i].z * e ;
    } 
}
''', 'kernel_test')

#-------------------------------------------------------
# PREPARAR LISTAS PARA LA GPU Y DEFINIR COLORES DE ARISTAS
#------------------------------------------------------- 
def vecinos_to_gpu(vecinos):
    indice_start = [0]
    vecinos_list = []

    for lista in vecinos:
        vecinos_list.extend(lista)
        indice_start.append(len(vecinos_list))

    return np.array(indice_start, dtype=np.int32), np.array(vecinos_list, dtype=np.int32)

# indice_start = [0, 3, 5]
# vecinos_list = [1,5,7,   0,2,   1,4,8,9]
indice_start, vecinos_list = vecinos_to_gpu(vecinos)
# listas para usar
indice_start_gpu = cp.asarray(indice_start)
vecinos_list_gpu = cp.asarray(vecinos_list)
# Definir proporciones de bloques/threads
threads_per_block = 256
num_vertices = pos.shape[0]
blocks = ( (num_vertices - 1) // threads_per_block ) + 1


#-------------------------------------------------------
# MOVER VÉRTICES MANUALMENTE O DEJARLOS FIJOS
#-------------------------------------------------------
def mover_vertice_manual(vertice_id, criterio_seleccion, direccion_movimiento, magnitud, 
                         modo_extremo, cantidad_extremos, 
                         modo_radial, radio_influencia):
    """
    Args:
        vertice_id (int): Vértice central/guía.
        criterio_seleccion (str): 'x+', 'z-', etc. (Solo para modo extremo/normal).
        direccion_movimiento (str): 'x+', 'z-', etc. Hacia dónde empujarlos.
        magnitud (float): Cuánto mover.
        modo_extremo (bool): Si es True, agarra los bordes del objeto.
        modo_radial (bool): Si es True, agarra una ESFERA alrededor del vertice_id.
        radio_influencia (float): (Solo modo radial) Qué tan grande es la esfera de selección.
    """
    global pos_gpu, vel_gpu, fue_gpu, indices_afectados
    
    indices_afectados = None
    
    # ---------------------------------------------------------
    # 1. DEFINIR EL VECTOR DE MOVIMIENTO (Hacia dónde van)
    # ---------------------------------------------------------
    vec_mov = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
    
    # Parsear direccion_movimiento
    eje_mov = 0
    signo_mov = 1
    if 'x' in direccion_movimiento: eje_mov = 0
    elif 'y' in direccion_movimiento: eje_mov = 1
    elif 'z' in direccion_movimiento: eje_mov = 2
    if '-' in direccion_movimiento: signo_mov = -1
    
    vec_mov[eje_mov] = magnitud * signo_mov

    # ---------------------------------------------------------
    # 2. SELECCIONAR INDICES
    # ---------------------------------------------------------
    
    # --- CASO A: MODO EXTREMO (Bordes) ---
    if modo_extremo:
        # Parsear criterio (ej: "x+")
        eje_sel = 0
        busqueda_positiva = True
        if 'x' in criterio_seleccion: eje_sel = 0
        elif 'y' in criterio_seleccion: eje_sel = 1
        elif 'z' in criterio_seleccion: eje_sel = 2
        if '-' in criterio_seleccion: busqueda_positiva = False
            
        indices_ordenados = cp.argsort(pos_gpu[:, eje_sel])
        
        if busqueda_positiva: 
            indices_afectados = indices_ordenados[-cantidad_extremos:]
        else:
            indices_afectados = indices_ordenados[:cantidad_extremos]
            
        #print(f"Extremo: Moviendo bordes en {criterio_seleccion}")

    # --- CASO B: MODO RADIAL (La "Bola" de influencia) ---
    elif modo_radial:
        try:
            lider_pos = pos_gpu[vertice_id] # Posición (x,y,z) en GPU
        except:
            return

        # 1. Calculamos vector distancia (Todos - Lider)
        diferencia_vec = pos_gpu - lider_pos 
        
        # 2. Calculamos la magnitud de ese vector (Distancia Euclideana)
        # axis=1 significa que suma x^2+y^2+z^2 por cada fila (vertice)
        distancias = cp.linalg.norm(diferencia_vec, axis=1)
        
        # 3. Filtramos los que estén DENTRO del radio
        indices_afectados = cp.where(distancias < radio_influencia)[0]
        
        #print(f"Radial: Agarrando {len(indices_afectados)} vértices cerca del ID {vertice_id}")

    # --- CASO C: MODO NORMAL (Rebanada plana) ---
    else:
        # Parsear criterio
        eje_sel = 0
        if 'x' in criterio_seleccion: eje_sel = 0
        elif 'y' in criterio_seleccion: eje_sel = 1
        elif 'z' in criterio_seleccion: eje_sel = 2
            
        try:
            lider_pos = cp.asnumpy(pos_gpu[vertice_id]) 
        except:
            return 

        val_lider = lider_pos[eje_sel]
        TOLERANCIA = 0.5
        
        diferencia = cp.abs(pos_gpu[:, eje_sel] - val_lider)
        indices_afectados = cp.where(diferencia < TOLERANCIA)[0]
        
        #print(f"Normal: Moviendo rebanada plana en eje {criterio_seleccion}")

    # ---------------------------------------------------------
    # 3. APLICAR MOVIMIENTO
    # ---------------------------------------------------------
    if indices_afectados is not None and len(indices_afectados) > 0:
        pos_gpu[indices_afectados] += vec_mov
        
        # Matar inercia
        vel_gpu[indices_afectados] = 0.0
        fue_gpu[indices_afectados] = 0.0
    
def fijar_borde_automatico(umbral_porcentaje):
    """
    Detecta automáticamente el borde exterior basándose en la distancia máxima.
    
    Args:
        umbral_porcentaje (float): 0.95 significa "Agarra todo lo que esté
                                   entre el 95% y el 100% de la distancia máxima".
                                   Si el borde es muy grueso, sube a 0.98.
                                   Si faltan puntos, baja a 0.90.
    """
    global pos_gpu, fijos_gpu, vel_gpu
    

    # 1. Calcular el centro promedio
    centro_objeto = cp.mean(pos_gpu, axis=0)
    
    # 2. Calcular distancias de TODOS los puntos al centro
    diferencia_vec = pos_gpu - centro_objeto
    distancias = cp.linalg.norm(diferencia_vec, axis=1)
    
    # 3. Encontrar la distancia MÁXIMA (el punto más lejano del objeto)
    max_dist = cp.max(distancias)
    
    # 4. Definir la distancia de corte (El límite para ser considerado "borde")
    distancia_corte = max_dist * umbral_porcentaje
    
    # 5. Seleccionar TODOS los vértices que superen esa distancia
    # "Si tu distancia es mayor que el 95% del máximo, eres borde"
    indices_afectados = cp.where(distancias >= distancia_corte)[0]
    
    # 6. Fijarlos
    if len(indices_afectados) > 0:
        fijos_gpu[indices_afectados] = 1
        vel_gpu[indices_afectados] = 0.0
    else:
        print("ALERTA: No se seleccionó ningún vértice. Revisa el umbral.")
    
    
#-------------------------------------------------------
# LOOP DE GRABACION
#------------------------------------------------------- 
# Decidir si video o no
if MODO_VIDEO:
    writer = imageio.get_writer(NOMBRE_VIDEO, fps=FPS)
else:
    print("no video")

# Empezar tiempo
tiempo_inicio = time.time()


for frame in range(TOTAL_FRAMES):
    indices_afectados = None
    fijar_borde_automatico(0.98)
    #---- ACTUALIZAR FUERZAS Y POSICIONES
    if frame > frame_espera:
        for _ in range(SUB_STEPS):
            # CALCULAR FUERZAS
            new_fuerzas((blocks,), (threads_per_block,), (fue_gpu, pos_gpu, indice_start_gpu, vecinos_list_gpu, 
                                                        np.int32(num_vertices), np.float32(k), np.float32(L0)))
            cp.cuda.Device().synchronize() 
        
            # MOVIMIENTOS DADOS
            if 30 <= frame < 50:
                # (ID, Criterio Seleccion, Direccion Movimiento, Magnitud, ...)
                mover_vertice_manual(vertice_id=15855, 
                                    criterio_seleccion="x-",     # SOLO PARA MODO EXTREMO Y MODO NORMAL
                                    direccion_movimiento="z+",   # HACIA DONDE MOVEREMOS
                                    magnitud=0.004,               
                                    modo_extremo=False,           #  MODO EXTREMO
                                    cantidad_extremos = 101*10,     # CANTIDAD EXTREMOS
                                    modo_radial=True,           # MODO RADIAL
                                    radio_influencia=1)          # RADIO PARA MODO RADIAL
            


            # CALCULAR POSICIONES
            new_pos((blocks,), (threads_per_block,), (fue_gpu, pos_gpu, vel_gpu, fijos_gpu,
                                                  np.int32(num_vertices), np.float32(m), np.float32(g), 
                                                  np.float32(damping), np.float32(e), np.float32(dt)))
            cp.cuda.Device().synchronize() 
    
    
    #---- TRAER LAS POSICIONES DE LA GPU A CPU
    pos = cp.asnumpy(pos_gpu)
    
    #---- RENDERIZAR Y GUARDAR
    if MODO_VIDEO:
        # Hacer fuerza de los vertices agarrados muy grande, para que se vean de color rojo
        if indices_afectados is not None:
            for i in (indices_afectados):
                fue_gpu[i] = [1000,0,0]
        
        # Definir colores dependiendo de las fuerzas
        magnitudes_gpu = cp.linalg.norm(fue_gpu, axis=1)
        intensidades_gpu = cp.clip(magnitudes_gpu / limite_fuerza, 0.0, 1.0)
        intensidades = cp.asnumpy(intensidades_gpu)
        colores = plt.cm.jet(intensidades)
        
        # Actualizar Aristas y sus colores 
        line.set_data(pos=pos[indices_lineas_np], color=colores[indices_lineas_np], connect='segments')
        #scatter.set_data(pos=pos, face_color=colores, size = TAMAÑO_PUNTOS, edge_width = 0)
        
        # Centrar cámara 
        view.camera.center = pos.mean(axis=0)
        
        # Poner frames en el video
        canvas.update()
        canvas.app.process_events()
        img = canvas.render(alpha=False)
        writer.append_data(img)
        
        
    #---- PROGRESO
    if frame % 10 == 0:
        print(f"Procesando frame {frame}/{TOTAL_FRAMES} ({(frame/TOTAL_FRAMES)*100:.1f}%)")

# Medicion tiempo   
cp.cuda.Device().synchronize()
tiempo_final = time.time()
duracion_total = tiempo_final - tiempo_inicio
print(f"{duracion_total:.5f}")
# Cerrar el archivo de video correctamente
if MODO_VIDEO:
    writer.close()
    print("¡Video guardado exitosamente!")





        
        




