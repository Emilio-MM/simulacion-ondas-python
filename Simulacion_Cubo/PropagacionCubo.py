import cupy as cp
import numpy as np
from vispy import app, scene
import matplotlib.pyplot as plt # Para el mapa de colores
vertice_seleccionado = None  # Numero de vertice
start_x_mouse = None 
start_y_mouse = None
grupo_seleccionado = None
lado_seleccionado = None

#-------------------------------------------------------
# CONFIGURACIONE DE VIDIO 
#-------------------------------------------------------
TAMAÑO_PUNTOS = 10
distancia_vista = 100
ruta = "cubo_125_5.obj"
limite_fuerza = 10.0 
frame_espera = 0

#-------------------------------------------------------
# CONSTANTES RESORTES 
#-------------------------------------------------------
L0 = 1        # longitud natural del resorte
k = 10.0      # constante del resorte
m = 0.05      # masa de cada punto
dt = 0.05     # delta de tiempo
damping = 0.9 # perdida de energia
e = 1.01       
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
# ARMAR ARISTAS COMO PARES DE PUNTOS
#-------------------------------------------------------
def definir_aristas(pos):
    vertices_par_posicion = []
    for i in range(len(caras_vuelta)):
        n_vertices = len(caras_vuelta[i])  # numero de vertices
        for j in range(n_vertices):        # recorre los vertices
            if j < n_vertices - 1:         # guarda dos pares de vertices
                v1 = caras_vuelta[i][j]
                v2 = caras_vuelta[i][j+1]
                vertices_par_posicion.extend([pos[v1], pos[v2]])
            
    return np.array(vertices_par_posicion)

#-------------------------------------------------------
# CARGA INICIAL 
#-------------------------------------------------------
pos, caras_vuelta = cargar_obj(ruta)  # Pos para cada vértice: [[x,y,z]], Caras para cada cara: [[p1,p2,p3,p4]]
vecinos = obtener_vecinos(caras_vuelta, len(pos))

# Inicializamos los arreglos de velocidades y fuerzas en 0 [[0, 0, 0]]
pos_gpu = cp.asarray(pos, dtype=cp.float32)
vel_gpu = cp.zeros((len(pos), 3), dtype=cp.float32)
fue_gpu = cp.zeros((len(pos), 3), dtype=cp.float32)

#-------------------------------------------------------
# VISPY 
#-------------------------------------------------------
# Crear la ventana/canvas
canvas = scene.SceneCanvas(keys='interactive', show=True)   
# Crear una "vista" dentro del canvas         
view = canvas.central_widget.add_view()      
# Cámara para interactuar                       
view.camera = scene.cameras.TurntableCamera(fov=45, distance=10)     
# Puntos
scatter = scene.visuals.Markers(pos=pos, size=8, face_color='red', parent=view.scene)
# Aristas
line = scene.visuals.Line(pos=definir_aristas(pos), color='white', width=2, connect='segments', parent=view.scene)
# Ejes
from vispy.scene.visuals import XYZAxis                              # dibujar ejes
from vispy.visuals.transforms import STTransform            
axis = XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(5, 5, 5), translate=(0, 0, 0))
view.camera.center = pos.mean(axis=0)                                # centar camara alrededor del objeto

#-------------------------------------------------------
# MOUSE MANIPULACION 
#-------------------------------------------------------
def seleccionar_cara(vertice, coords_screen):
    global grupo_seleccionado, lado_seleccionado
    
    # tamaño ventana
    W, H = canvas.size
    cx, cy = coords_screen  # posición del click en pantalla
    
    # puntos de referencia
    ref_top    = np.array([W/2, 0])
    ref_bottom = np.array([W/2, H])
    ref_left   = np.array([0,   H/2])
    ref_right  = np.array([W,   H/2])
    
    # distancias
    d_top    = np.linalg.norm(np.array([cx,cy]) - ref_top)
    d_bottom = np.linalg.norm(np.array([cx,cy]) - ref_bottom)
    d_left   = np.linalg.norm(np.array([cx,cy]) - ref_left)
    d_right  = np.linalg.norm(np.array([cx,cy]) - ref_right)
    
    distancias = [d_top, d_bottom, d_left, d_right]
    lado_seleccionado = np.argmin(distancias)   # 0=arriba, 1=abajo, 2=izquierda, 3=derecha
    
    # ahora según el lado, decidir qué cara agarrar
    if lado_seleccionado == 0:   # ARRIBA → cara +Y
        umbral = np.max(pos[:,2]) - 0.01
        grupo_seleccionado = [i for i,p in enumerate(pos) if p[2] >= umbral]

    elif lado_seleccionado == 1: # ABAJO → cara -Y
        umbral = np.min(pos[:,2]) + 0.01
        grupo_seleccionado = [i for i,p in enumerate(pos) if p[2] <= umbral]

    elif lado_seleccionado == 2: # IZQUIERDA → cara -X
        umbral = np.min(pos[:,0]) + 0.01
        grupo_seleccionado = [i for i,p in enumerate(pos) if p[0] <= umbral]

    elif lado_seleccionado == 3: # DERECHA → cara +X
        umbral = np.max(pos[:,0]) - 0.01
        grupo_seleccionado = [i for i,p in enumerate(pos) if p[0] >= umbral]
      
#-------------------------------------------------------
# MOUSE MANIPULACION 
#-------------------------------------------------------
# Funcion para seleccionar un vertice cercano
@canvas.events.mouse_press.connect 
def mouse_press(event): 
    global vertice_seleccionado, start_x_mouse, start_y_mouse

    # proyectar todos los vertices a coordenadas de pantalla 
    verts_screen = view.scene.transform.map(pos)                     # Fase previa para hacer perspectiva (x´,y´,z´,w´)
    verts_screen = verts_screen[:, :2] / verts_screen[:, 3][:, None] # solo toma x´ , y´   / solo toma w´ y la convierte en columna Nx1  
    # coordenadas del mouse en pantalla (x,y)
    mx, my = event.pos
    # distancia entre el mouse y cada vertice en pantalla
    dx = verts_screen[:, 0] - mx # toma de cada par de valor solo el primero
    dy = verts_screen[:, 1] - my # toma de cada par de valor solo el segundo
    dist = np.sqrt(dx**2 + dy**2)

    # obtener los vertices mas cercanos a los clickados
    dist_min = np.min(dist)
    if dist_min <= 5:                             # solo si sí lo picas se guardara
        vertice_seleccionado = np.argmin(dist)    # devuelve el indice del vertice mas cercano
        start_x_mouse, start_y_mouse = event.pos  # guarda la posicion de comienzo del mouse x,y
        coords = event.pos                        # (x,y) del click en pantalla
        seleccionar_cara(vertice_seleccionado, coords)
        
        view.camera.interactive = False           # bloquea la camara
        print("vértice:", vertice_seleccionado)            

# Funcion para que el vertice siga el movimiento del mouse
@canvas.events.mouse_move.connect  
def mouse_move(event):  
    global pos, start_x_mouse, start_y_mouse

    if vertice_seleccionado is None:
        return
    
    # posicion actual del mouse arrastrado
    mx, my = event.pos     
    # diferencias en pixeles del arrastre
    dx = mx - start_x_mouse
    dy = -my + start_y_mouse
    # esa diferencia sumarla a la posicion de los vertices seleecionados
    indices = np.array(grupo_seleccionado)
    # SEGÚN EL LADO SELECCIONADO, MOVER EN EL EJE CORRECTO
    if lado_seleccionado == 0:     # ARRIBA  (+Y)
        pos[indices, 2] += dy * 0.01

    elif lado_seleccionado == 1:   # ABAJO   (-Y)
        pos[indices, 2] += dy * 0.01

    elif lado_seleccionado == 2:   # IZQUIERDA (-X)
        pos[indices, 0] += dx * 0.01

    elif lado_seleccionado == 3:   # DERECHA (+X)
        pos[indices, 0] += dx * 0.01
    # actUalizar el pos_gpu
    pos_gpu[indices] = cp.asarray(pos[indices])
    # actualizar origen para el siguiente frame
    start_x_mouse = mx
    start_y_mouse = my
    
@canvas.events.mouse_release.connect 
def mouse_release(event):
    global vertice_seleccionado, grupo_seleccionado
    vertice_seleccionado = None 
    grupo_seleccionado = None
    view.camera.interactive = True             # desbloquea la camara
    
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
    int num_vertices,
    float m,
    float g,
    float damping,
    float e,
    float dt
){
                                                     // cantidad_threads_por_bloque * idx_bloque_en_el_grid + idx_hilo_en_su_bloque
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // indice global
    if (i >= num_vertices) return;

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
    if (pos[i].z < 0){ 
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

def definir_colores_aristas(colores_vertices):
    colores_para_lineas = []
    # Usamos la misma logica que usaste para las posiciones
    for i in range(len(caras_vuelta)):
        n_vertices = len(caras_vuelta[i])
        for j in range(n_vertices):
            if j < n_vertices - 1:
                v1 = caras_vuelta[i][j]
                v2 = caras_vuelta[i][j+1]
                # Agregamos el color del inicio y el color del final
                colores_para_lineas.extend([colores_vertices[v1], colores_vertices[v2]])
                
    return np.array(colores_para_lineas)

#-------------------------------------------------------
# LOOP DE ANIMACION
#------------------------------------------------------- 

def update(event):
    global pos
    
    # DETENER EL CALCULO DE LOS VERTICES SELECCIONADOS
    if vertice_seleccionado is not None and grupo_seleccionado is not None:
        # Convertimos la lista de python a cupy array para indexar rapido
        indices_gpu = cp.array(grupo_seleccionado, dtype=cp.int32)
        vel_gpu[indices_gpu] = 0.0
        fue_gpu[indices_gpu] = 0.0
        
    # CALCULAR LAS FUERZAS MODIFICANDO FUE_GPU
    new_fuerzas((blocks,), (threads_per_block,), (fue_gpu, pos_gpu, indice_start_gpu, vecinos_list_gpu, 
                                                  np.int32(num_vertices), np.float32(k), np.float32(L0)))
    cp.cuda.Device().synchronize() 
    
    # CALCULAR LAS POSICIONES MODIFICANDO POS_GPU
    new_pos((blocks,), (threads_per_block,), (fue_gpu, pos_gpu, vel_gpu,
                                              np.int32(num_vertices), np.float32(m), np.float32(g), 
                                              np.float32(damping), np.float32(e), np.float32(dt)))
    cp.cuda.Device().synchronize() 
    
    # TRAER LOS DATOS DELA GPU
    pos = cp.asnumpy(pos_gpu) 
    
    # COLOREADO DE VERTICES BASADO EN FUERZA
    magnitudes_gpu = cp.linalg.norm(fue_gpu, axis=1)
    intensidades_gpu = cp.clip(magnitudes_gpu / limite_fuerza, 0.0, 1.0)
    intensidades = cp.asnumpy(intensidades_gpu)
    colores = plt.cm.jet(intensidades)

    # COLOREADO DE ARISTAS 
    vertices_par_posicion = definir_aristas(pos)
    colores_par_aristas = definir_colores_aristas(colores)       
    line.set_data(pos=vertices_par_posicion, color=colores_par_aristas, connect='segments')   
    scatter.set_data( pos=pos, face_color=colores,size=TAMAÑO_PUNTOS)
          
    view.camera.center = pos.mean(axis=0) # centar camara alrededor del objeto  

timer = app.Timer()
timer.connect(update)
timer.start(0.05) # corre la funcion update cada 0.05s
app.run()


        
        




