
# Simulaciones Físicas con Aceleración por GPU (CUDA)
Este repositorio contiene un entorno de simulación interactiva diseñado para analizar la dinámica de cuerpos blandos y membranas elásticas mediante sistemas de masa-resorte. El núcleo del proyecto utiliza kernels de CUDA escritos en C++ y ejecutados a través de CuPy para procesar miles de cálculos de fuerzas y posiciones en paralelo, permitiendo simulaciones fluidas en tiempo real o generación de video de alta fidelidad.

## Estructura del Proyecto
El repositorio se divide en dos secciones principales, cada una ubicada en su propia carpeta:

Simulacion del Cubo: Enfocada en la manipulación manual. Permite interactuar con un objeto 3D (cubo) utilizando el mouse para agarrar vértices y observar la propagación de ondas mecánicas.

Simulacion de Membranas: Orientada al análisis de membranas (como tambores o telas). Incluye funciones de fijado automático de bordes y modos de movimiento radial o por rebanadas para simular impactos o vibraciones controladas.

## Cada carpeta incluye:
-El código fuente en Python (.py).    
-Un archivo de malla geométrica (.obj) necesario para la ejecución.  
-Videos de ejemplo (.mp4 / .gif) que muestran el comportamiento esperado del programa.

## Requisitos Técnicos
Debido a que el cálculo de físicas se delega a la tarjeta gráfica, es indispensable contar con el siguiente hardware y software:  
-GPU: Tarjeta NVIDIA compatible con la arquitectura CUDA (probado en RTX 3050 Ti).  
-Software: NVIDIA CUDA Toolkit instalado en el sistema.  

## Librerías de Python:  
-cupy: Procesamiento en paralelo en GPU.  
-vispy: Renderizado y visualización 3D interactiva.  
-numpy: Manejo de datos y arreglos.  
-matplotlib: Generación de mapas de colores térmicos.  
-imageio: Grabación y exportación de video (requerido para el modo video).  

## Configuración y Personalización  
El comportamiento de los materiales se puede ajustar modificando las variables dentro de la sección de CONSTANTES RESORTES en cada script:  
-k: Constante de rigidez del resorte (determina qué tan "duro" es el material).  
-m: Masa de cada punto (influye en la inercia del objeto).  
-damping: Factor de amortiguamiento (pérdida de energía por fricción interna).  
-L0: Longitud natural de los resortes (distancia de reposo entre puntos).  
-e: Coeficiente de restitución (constante de rebote contra el suelo).  

## Instrucciones de Uso
Para ejecutar cualquiera de las simulaciones, asegúrate de que el archivo .obj correspondiente esté en la misma carpeta que el código.

## Simulación del Cubo (Interacción Manual)
1.Ejecuta el programa. Se abrirá una ventana de visualización.  
2.Cámara: Usa el clic izquierdo en el fondo para rotar la perspectiva.  
3.Manipulación: Haz clic sobre un vértice o cara del cubo y arrastra el mouse. El código detectará el punto más cercano en pantalla, calculará la deformación y activará la respuesta elástica al soltarlo.  


## Simulación de Membranas (Modo Grabación/Automático)
![Membrana radial](Propagacion_Membrana/demo-simulacion-membrana.gif)

Este código está configurado para exportar un archivo de video llamado simulacion_resortes.mp4.

Fijado de bordes: Utiliza la función fijar_borde_automatico que detecta los límites exteriores de la malla y los mantiene estáticos, ideal para simular parches de percusión.

Modo de selección: Permite aplicar fuerzas de tres formas:

Normal: Selecciona una rebanada plana del objeto.

Extremo: Selecciona bordes específicos en los ejes X, Y o Z.

Radial: Afecta a un grupo de vértices dentro de una esfera de influencia alrededor de un punto central.

Análisis Visual
El sistema incluye un mapa de calor dinámico. Los vértices y aristas cambian de color (del azul al rojo) en tiempo real según la magnitud de la fuerza elástica acumulada en esa zona, permitiendo identificar puntos de máxima tensión mecánica durante la vibración.
