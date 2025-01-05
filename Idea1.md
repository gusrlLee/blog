<!-- naming rule : Author's last name (YYYY), "Title"  -->
* Global Illumination 
  * Photon Mapping 
    * **Jensen (1996), "Global illumination using photon maps."**
      * Takeaways
        * Two-pass global illumination 
          * Constructing the Photon maps
          * Rendering by using the Photon maps
        * Comparisons with existing global illumination (ex. Radiosity) techniques indicate that the photon map provides an efficient environment for global illumination.
      * Paper contexts
        * Introduction 
          * Proposed problem : 
          In general Monte Carlo ray tracing is very time consuming 
          and gives noisy results while radiosity uses a lot of memory to store directional information 
          and it cannot handle specular reflection properly.
          * Their motivation (여기서 Radiosity method에 대한 이야기) was the fact that radiosity becomes very time 
          and memory consuming as the number of surfaces in the model grows
          * This paper introduces a two pass method in which we simplify the representation of the illumination instead of simplifying the geometry.
          * The photon map is used to generate optimizied sampling directions, to reduce the numeber of shadow rays,
          to render caustic and to limit the number of reflections traced as the scene is rendered with distribution ray tracing.
        * Overview of the method
          * Our motivation for doing so is that we obtain a very flexible environment with a lot of useful information that can be applied in the rendering step. 
          * Pass 1 : Constructing the Photon Maps
            * Unlike previous implementations we use two photon maps:
              * A caustics photon map 
                * This is used only to store photons corresponding to caustics and it is created by emitting photons 
                towards the specular objects in the scene and storing these as they hit diffuse surfaces.
              * A global photon map 
                * This is used as a rough approximation of the light/flux within the scene and it is created by emitting photons towards all objects.
                * We create shadow photons by tracing rays with origin at the light source through the entire scene. 
          * Pass 2 : Rendering 
            * The final image is rendered using Monte Carlo ray tracing in which the pixel radiance is computed by averaging a number of sample estimates.
            * $L_s (x, \Psi_r)$ is computed using the rendering equation.
              * $L_s(x, \Psi_r) = L_e (x, \Psi_r) + \int_\Omega f_r(x, \Psi_i; \Psi_r) L_i (x, \Psi_i)\cos \theta_i d\omega_i \to (1)$
                * $L_r$ is radiance emitted by the surface 
                * $L_i$ is the incoming randiance in the direction $\Psi_i$
                * $f_r$ is the BRDF 
                * $\Omega$ is the sphere of incoming directions
            * The radiance returned by each ray is computed at the first surface intersected by the ray.
            * The rendering eqaution (1) can be split into a sum of several components. 
              * $L_r = \int_\Omega f_r L_{i,l} \cos\theta_i d \omega_i + \int_\Omega f_{r,s} (L_{i,c} + L_{i,d})\cos\theta_i d \omega_i + \int_\Omega f_{r, d} L_{i,c} \cos\theta_i d \omega_i + \int_\Omega f_{r,d} L_{i,d} \cos\theta_i d \omega_i$
                * $f_r = f_{r, s} + f_{r, d} \;\; and \;\; L_i = L_{i, l} + L_{i, c} + L_{i, d}$
                * $L_{i,l}$, the incoming radiance has been split into contributions from the light sources
                * $L_{i,c}$, contributions from the light sources via specular reflection (caustics)
                * $L_{i, d}$, indirect soft illumination
                * $f_{r,d}$, the BRDF has been separated into a diffuse part
                * $f_{r, s}$, a specular part
          * Estimating Radiance using the Photon Map
            * we can integrate the information into the rendering equation as follows:  
            * $L_r(x, \Psi_r) = \int_\Omega f_r(x, \Psi_r, \Psi_i) \frac{d^2 \Phi_i(x, \Psi_i)}{dA d\omega_i} d\omega_i \approx \sum^N_{p = 1} f_r(x, \Psi_r, \Psi_{i,p}) \frac{\Delta\Psi_p(x, \Psi_{i,p})}{\pi r^2}$ 
            * 기존 Monte Carlo estimation 과 매우 유사
    * **Hachisuka et al. (2008), "Progressive pßhoton mapping"**
    * **Hachisuka and Jensen (2009), "Stochastic progressive photon mapping."**
    * **Hachisuka and Jensen (2010), "Parallel progressive photon mapping on GPUs."**
    * **Knaus (2011), "Progressive photon mapping: A A probabilistic approach."**
    * **Tokuyoshi and Jensen (2011), "Robust Adaptive Photon Tracing using Photon Path Visibility"**
    * **Georgiev et al. (2012), "Light transport simulation with vertex connection and merging."**
    * **Mara et al. (2013), "Toward practical real-time photon mapping: Efficient GPU density estimation."**
    * **Davidovič et al. (2014), "Progressive light transport simulation on the GPU: Survey and improvements"**
    * **Evangelou et al. (2021), "Fast radius search exploiting ray tracing frameworks."**
    * **Kern et al. (2023), "Accelerated photon mapping for hardware-based ray tracing."**
    * **Kim et al. (2019), "Caustics using screen-space photon mapping." -> Ray tracing Gems I book**
    * **Yang and Ouyang (2021), "Real-time ray traced caustics." -> Ray tracing Gems II book**

  * Reservoir Resampling. 
    * **Chao (1982), "A general purpose unequal probability sampling plan."**
    * **Vitter (1985), "Random sampling with a reservoir."**
    * **Talbot et al. (2005), "Importance Resampling for Global Illumination."**
    * **Majercik et al. (2019), "Dynamic diffuse global illumination with ray-traced irradiance fields."**
    * **Bitterli et al. (2020), "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting."**
    * **Wyman and Panteleev (2021), "Rearchitecting Spatiotemporal Resampling for Production."**
    * **Bokšanský et al. (2021), "Rendering Many Lights with Grid-Based Reservoirs."**
    * **Ouyang et al. (2021), "ReSTIR GI: Path resampling for real time path tracing."**
    * **Boissé (2021), "World-space spatiotemporal reservoir reuse for ray traced global illumination."**
    * **Majercik et al. (2021), "Scaling probe-based real-time dynamic global illumination for production."**
    * **Majercik et al. (2022), "Dynamic diffuse global illumination resampling."**
    * **Lin et al. (2022), "Generalized resampled importance sampling: Foundations of ReSTIR."**
    * **Tokuyoshi (2023), "Efficient spatial resampling using the pdf similarity."**
    * **Wyman et al. (2023), "A gentle introduction to restir path reuse in real-time."**
    * **Kettunen et al. (2023), "Conditional resampled importance sampling and ReSTIR."**
    * **Sawhney et al. (2024), "Decorrelating restir samplers via mcmc mutations."**
    * **Zhang et al (2024), "Area ReSTIR: Resampling for Real-Time Defocus and Antialiasing"**

  * Quantization (we have to find more reference about this mehtod.)
    * **Schütz et al. (2021), "Rendering point clouds with compute shaders and vertex order optimization."**
    * **Schütz et al. (2022), "Software Rasterization of 2 Billion Points in Real Time"**
    * **Schuster et al. (2021), "Compression and Rendering of Textured Point Clouds via Sparse Coding."**
  
  * Ray coherence & scheduling (we have to find more reference about this mehtod.)
    * **Navrátil et al. (2007), "Dynamic ray scheduling to improve ray coherence and bandwidth utilization"**
    * **Lee et al. (2017), "Vectorized Production Path Tracing"**