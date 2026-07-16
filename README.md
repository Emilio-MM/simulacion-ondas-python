# GPU-Accelerated Physics Simulations (CUDA)

This repository contains an interactive simulation environment designed to analyze the dynamics of soft bodies and elastic membranes using mass-spring systems. The core of the project utilizes CUDA kernels written in C++ and executed through CuPy to process thousands of force and position calculations in parallel, enabling smooth real-time simulations or high-fidelity video generation.

<div align="center">
  <img src="Propagación-Membrana/demo-simulacion-membrana.gif" width="250" />
  <img src="Propagación-Membrana/demo-simulacion-membrana-ejex.gif" width="400" />
</div>

## Project Structure
The repository is divided into two main sections, each located in its own folder:

* **Cube Simulation:** Focused on manual manipulation. It allows interacting with a 3D object (cube) using the mouse to grab vertices and observe the propagation of mechanical waves.
* **Membrane Simulation:** Oriented toward the analysis of membranes (like drums or fabrics). It includes functions for automatic edge pinning and radial or slice movement modes to simulate controlled impacts or vibrations.

### Each folder includes:
* The Python source code (`.py`).  
* A geometric mesh file (`.obj`) required for execution.  
* Example videos (`.mp4` / `.gif`) showing the expected behavior of the program.

## Technical Requirements
Since physics calculations are delegated to the graphics card, the following hardware and software are essential:  
* **GPU:** NVIDIA card compatible with CUDA architecture (tested on an RTX 3050 Ti).  
* **Software:** NVIDIA CUDA Toolkit installed on the system.  

### Python Libraries:  
* `cupy`: GPU parallel processing.  
* `vispy`: Interactive 3D rendering and visualization.  
* `numpy`: Data and array handling.  
* `matplotlib`: Thermal color map generation.  
* `imageio`: Video recording and export (required for video mode).  

## Configuration and Customization  
The behavior of the materials can be adjusted by modifying the variables within the `SPRING CONSTANTS` section in each script:  
* `k`: Spring stiffness constant (determines how "hard" the material is).  
* `m`: Mass of each point (influences the object's inertia).  
* `damping`: Damping factor (energy loss due to internal friction).  
* `L0`: Natural spring length (resting distance between points).  
* `e`: Coefficient of restitution (bounce constant against the ground).  

## Usage Instructions
To run any of the simulations, ensure the corresponding `.obj` file is in the same folder as the code.

### Cube Simulation (Manual Interaction)
<div align="center">
  <img src="Propagacion-Cubo/demo-simulacion-cubo.gif" width="500" />
</div>

1. Run the program. A viewing window will open.  
2. **Camera:** Use left-click on the background to rotate the perspective.  
3. **Manipulation:** Click on a vertex or face of the cube and drag the mouse. The code will detect the closest point on the screen, calculate the deformation, and trigger the elastic response upon release.  

### Membrane Simulation (Recording/Automatic Mode)
<div align="center">
  <img src="Propagación-Membrana/demo-simulacion-lamina.gif" width="500" />
</div>

This code is configured to export a video file named `simulacion_resortes.mp4`.

* **Edge pinning:** Uses the `fijar_borde_automatico` function, which detects the outer limits of the mesh and keeps them static, ideal for simulating percussion drumheads.
* **Selection mode:** Allows applying forces in three ways:
  * *Normal:* Selects a flat slice of the object.
  * *Extreme:* Selects specific edges on the X, Y, or Z axes.
  * *Radial:* Affects a group of vertices within a sphere of influence around a central point.

## Visual Analysis
The system includes a dynamic heatmap. Vertices and edges change color (from blue to red) in real-time based on the magnitude of the accumulated elastic force in that area, allowing the identification of points of maximum mechanical stress during vibration.
