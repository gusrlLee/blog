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
      * Takeaways
        * Prgressive photon mapping is a multi-pass algorithm:
          * first pass : Ray tracing pass 
          * second pass : All subsequent passes use photon tracing pass 
        * PPM show that progressive photon mapping is particularly robust in scenes with complex caustics illumination, 
        and it is more efficient than methods based on Monte Carlo ray tracing such as BPT, MLT.
      * Paper contexts 
        * Introduction 
          * Monte Carlo based methods can simulate both specular and diffuse materials, 
          but there is one combination of these materials that is particularly problematic for most of the methods.
          This combination involves light being transported along a specular to diffuse to specular path (SDS path) before being seen by the eye.
          * SDS paths are particularly challenging when the light source is small since the probability of sampling the light 
          through the specular material is low in unbiased Monte Carlo ray tracing methods such as PT, BPT, MLT.
          * Proposed problem : 
          Photon mapping is a consistent algorithm which is good at simulating caustics and SDS paths. 
          However, photon mapping becomes very costly for scenes dominated by caustics illumination since the caustics are simulated by directly visualizing the photon map.
          * To avoid noise it is necessary to use a large number of photons 
          and the accuracy is limited by the memory avaiable for the photon map.
          * In this paper we present a progressive photon mapping algorithm that makes it possible to robustly simulate global illumination 
          including SDS paths with arbitrary accuarcy without requiring infinite memory.
        * Progressive Photon Mapping 
          * The main idea in progressive photon mappng is to reorganize the standard photon mapping algorithm based on the conditions of consistency,
          in order to compute a global illumination solution with arbitrary accuracy without storing the full photon map in memory.
          * Prgressive photon mapping is a multi-pass algorithm:
            * first pass : Ray tracing pass 
              * It uses standard ray tracing to find all the surfaces in the scene visible through each pixel in the image (or a set of pixels).
              * Note, that each ray path includes all specular bounces until the first non-specular surface seen.
            * second pass : All subsequent passes use photon tracing pass 
              * This step is used to accumulate photon power at the hit points found in the ray tracing pass.
              * Once the contribution of the photons have been recorded they are no longer needed, 
              and we can discard all photons and proceed with a new photon tracng pass.
        * Progressive Radiance Estimation 
          * The key insight that makes this possible is a new technique for reducing the radius in the radiance estimate at each hit point,
          while increasing the number of accumulated photons.
          * Radius Reduction 
            * new photon density = $\hat{d}(x) = \frac{N(x) + M (x)}{\pi R(x)^2}$
              * $N(x)$ : the number of photon accumulated within this radius
              * $M(x)$ : Find $M(x)$ photons within the radius $R(x)$
              * $\hat{N} = N(x) + \alpha M(x)$
              * $\hat{R}(x) = R(x) - d R(x) = R(x) \sqrt{\frac{N(x) + \alpha M(x)}{N(x) + M(x)}}$
          * Flux Correction 
            * $N(x) \to \tau_N(x, \vec{\omega}) = \sum^{N(x)}_{p=1} f_r (x, \vec{\omega}, \vec{\omega}_p) \phi_p'(x_p, \vec{\omega}_p)$
            * $M(x) \to \tau_M(x, \vec{\omega}) = \sum^{M(x)}_{p=1} f_r (x, \vec{\omega}, \vec{\omega}_p) \phi_p'(x_p, \vec{\omega}_p)$
            * $\tau_{\hat{N}} (x, \vec{\omega}) = \tau_{N+M}(x, \vec{\omega}) \frac{N(x) + \alpha M(x)}{N(x) + M(x)}$
            * $\tau_{N+M} = \tau_N(x, \vec{\omega}) + \tau_M(x, \vec{\omega})$
            * $\tau_{\hat{N}} (x, \vec{\omega})$ is the reduced value for the reduced radius disc corresponding to $\hat{N}(x)$ photons.
          * Radiance Evaluation 
            * The radiance is evaluated as follows :
              * $L(x, \vec{\omega}) = \frac{1}{\pi R(x)^2} \frac{\tau(x, \vec{\omega})}{N_{emitted}}$
    * **Hachisuka and Jensen (2009), "Stochastic progressive photon mapping."**
      * Takeaways
        * We have presented a new formulation of progressive photon mapping, called stochastic progressive photon mapping, 
        that makes it possible to compute the correct average radiance value over a region.
        * new formulation : progressive photon mapping by adding a new distributed ray tracing pass that generate new hit points 
        after each photon pass.
      * Paper contexts
        * Introduction 
          * The results of PM suffer from bias, which appears as low frequency noise in the rendered images. 
          Moreover, computing the correct solutions requires storing an infinite number of photons in the limits.
          * Progressive photon mapping (PPM) solves this issue by using progressive refinement,
          and makes it possible to compute a correct solutions without storing any photons.
          Moreover, the method retains the robustness of photon mapping.
          * Althought each radiance estimate in progressive photon mapping converges to the correct radiance, 
          the algorithm is restricted to computing the correct radiance value at a point. 
          This property limits the application of progressive photon mapping because we often need to compute the correct average radiance value over a region.
          * In this paper, we present a new formulation of progressive photon mapping that enables computing the correct average radiance value over a region.
        * Overview 
          * Stochasitc Progressive Photon Mapping (SPPM)
            * The motivation is that we need to compute the average radiance value over a region in order to render distributed ray tracing effects.
            * Our idea is to use shared statistics over a region that we would like to compute the average radiance value for.
            * the average radiance value $L(S, \vec{\omega}) \approx \frac{\tau_i(S, \vec{\omega})}{N_e(i)\pi R_i(S)^2}$
              * $\tau_i{S, \vec{\omega}}$ is the shared accumulated flux over the region $S$
              * $R_i(S)$ is the shared search radius
            * The updating procedure of the shared statistics is 
              * $\vec{x}_i$ : is a randomly generated position within $S$ and $N_i(S)$ is the shared local photon count.
              * $N_{i+1} = N_i(S) + \alpha M_i(\vec{x}_i)$
              * $R_{i+1}(S) = R_i(S) \sqrt{\frac{N_i(S) + \alpha M_i(\vec{x}_i)}{N_i(S) + M_i(\vec{x}_i)}}$
              * $\Phi_i (\vec{x}_i, \vec{\omega}) = \sum^{M_i(\vec{x}_i)}_{p=1}f_r(\vec{x}_i, \vec{\omega}, \vec{\omega}_p)\Phi_p(\vec{x}_p, \vec{\omega}_p)$
              * $\tau_{i+1} (S, \vec{\omega}) = (\tau_i(S, \vec{\omega}) + \Phi_i(\vec{x}_i, \vec{\omega})) \frac{R_{i+1}(S)^2}{R_i(S)^2}$
            * Stochastic Radiance Estimate 
              * Our formulation assumes that the initial radius $R_0$ is constant within $S$,
              and the value of $\alpha$ is also constant within $S$.
              * $R_{i+1}(\vec{x}) = R_i(\vec{x})C_p$
            * Sec 4.1 Shared Radius and Sec 4.2 Shared Accumulated Flux are derivation.

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

  * CPU-GPU Hybrid method
    * **Barringer et al. (2017), "Ray Accelerator: Efficient and Flexible Ray Tracing on a Heterogeneous Architecture"**