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
    * **Hachisuka et al. (2008), "Progressive photon mapping"**
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
      * Takeaways
        * We present a data-parallel implementation of progressive photon mapping on GPUs.
            * The key contribution is a new stochastic spatial hashing scheme that achieves a data-parallel construction of a photon map and an efficient range query of a photon map on GPUs.
            * i) hash, ii) grid
      * Paper contexts 
        * Introduction 
          * However, an efficient implementation of progressive photon mapping on GPUs poses some challenges.
          * In particular, we need a fast construction of photon maps and 
          efficient range query of photons on GPUs since the algorithm repeatedly constructs and uses a photon map.
          * We present a data-parallel implementation of progressive photon mapping on GPUs.
            * The key contribution is a new stochastic spatial hashing scheme that achieves a data-parallel construction of a photon map and an efficient range query of a photon map on GPUs.
        * Method 
          * Our implementation uses spatial hashing for photon maps.
          * Creating a list is not suitable for GPUs because this is a dependent and serial process.
            * Our solution is to stochastically store a single photon instead of storing a list photons at each hash entry.
          * More precisely, given $n$ photons that would be mapped to the same hash entry, 
          we select one photon according to the uniform probability $p(x) = \frac{1}{n}$. 
            * To keep the result consistent, stored photon flux is now divided by the probability $p(x) = \frac{1}{n}$.
            * we can ignore hash collisionn.
          * We can also perform range query of photons just by looking at grid cells that intersect with a sphere 
          defined by query position and query radius.
    * **Knaus (2011), "Progressive photon mapping: A probabilistic approach."**
      * Takeaways
        * PPM의 bias를 없애기 위한 probabilistic approach를 적용
        * PM에서 가능한 variance와 expected 에 대한 수식 가지고 있음
        * $\bar{c}_N = \frac{1}{N}\sum^N_{i=1} \frac{1}{p_e(x_i, \omega_i)} W(x_i, \omega_i)(L(x_i,\omega_i) + \epsilon_i). \to Eq. (7)$
      * Paper contexts
        * Introduction 
          * One of the main advantage of photon mapping is that, at equal computational cost, 
          it can often produce images with less noise than other Monte Carlo algorithms.
          * However, it is biased, which means that the expected error of any approximation with a limited number of samples is nonzero.
          * In this article, we introduce a probabilistic derivation of progressive photon mapping.
            * The key property of our approach is that it does not require the maintenance of local statistic.
            Therefore, we could call our approach memoryless progressive photon mapping. $\to$ MPPM
        * Problem formulation 
          * We use a probabilistic framework that does not rely on assumptions about a specific kernel in the photon radiance estimate.
          It also does not rely on the maintenance of local statistics.
          * Our goal is to compute pixel value $c$ of the form : $c = \int W(x, \omega)L(x, \omega)dxdw \to Eq. (6)$
          * 4.1 Monte Carlo Approximation and Photon Maps (Paper와 같이 읽기)
            * A main issue in conventional photon mapping is that there is a trade-off between the variance, 
            or noise, and the expected error, or bias, in the radiance estimate.
            * One can either achieve a low variance or a low expected error, but not both.
            * The main insight of progressive photon mapping is that we can obtain a solution with, in the limit,
            vanishing variance and expected error by averaging the results over many photon maps.
            * To explain how this is possible
              * the evaluation of Eq. (6) as a Monte Carlo estimation 
              $c = \int W(x, \omega)L(x, \omega)dxdw. \to Eq. (6)$
              * we omit the arguments of the error using the shorthand $\epsilon_i = \epsilon(x, r_i)$.
              Using this model, our Monte Carlo estimate is :
              $\bar{c}_N = \frac{1}{N}\sum^N_{i=1} \frac{1}{p_e(x_i, \omega_i)} W(x_i, \omega_i)(L(x_i,\omega_i) + \epsilon_i). \to Eq. (7)$
                * $\bar{c}_N$ denotes the estimate for a pixel value using $N$ samples, that is, eye paths.
                * The sample $(x_i, \omega_i)$ denote position and direction of surface intersections of paths generated by ray tracing starting form the eye, 
                their probabilty density is $p_e(x_i, \omega_i)$
              * $\bar{\epsilon}_N = \frac{1}{N} \sum^N_{i=1} \epsilon_i. \to Eq. (8)$
              * $as N \to \infty \to Eq. (9)$
                * $Var[\bar{\epsilon}_N] \to 0 \Rightarrow Var[\bar{c}_N] \to 0$
                * $E[\bar{\epsilon_N} \to 0 \Rightarrow E[\bar{c}_N] \to 0]$
          * Achieving Convergence
            * The main idea of progressive photon mapping is to let the variance of the radiance estimate inicreate by a small factor in each step, but such that in average it still vanishes.
            * Variance 
              * $Var[\epsilon(x, r)] \approx \frac{(Var[\gamma] + E[\gamma]^2)p_l(x)}{Mr^2}\int_{\mathbb{R}^2} k(\psi)^d\psi$
            * Exepected Error
              * $E[\epsilon(x, r)] = r^2E[\gamma]\tau$
            * 증명이 작성되어 있는 Sec.
        * Algorithm
          * 1. generate photon map once
          * 2. trace path from eye until diffuse surfac is hit
          * 3. hit position and direction are $(x_i, \omega_i)$
          * 4. path contribution is $W(x_i, \omega_i)$
          * 5. path probability density is $p_e(x_i, \omega_i)$
          * 6. get current radius $r_i$ from reference $r_1$ and Eq. 15
          * 7. obtain radiance estimate $L(x_i, \omega_i) + \epsilon_i$
          * 8. update pixel value, Eq. 7
          * 9. iteration about all pixels (2-9)
    * **Tokuyoshi and Jensen (2011), "Robust Adaptive Photon Tracing using Photon Path Visibility"**
      * Takeaways
        * We present a new adaptve photon tracing algorithm which can handle illumination settings 
        that are considered difficult for photon tracing approaches such as outdoor scenes, close-ups of a small part of 
        an illuminated region, and illumination coming through a samll gap.
        * The key contribution in our algorithm is the use of visibility of photon path as the importance function which
        ensures that our sampling algorithm focuses on paths that are visible from the given viewpoint.
          * Adaptive Markov chain Monte Carlo methods + Replica exchange Monte Carlo method
      * Paper contexts 
        * Introduction
          * Progressive photon mapping can handle specular-diffuse-specular light transport robustly, however, it becomes 
          inefficient in scenes where only a samll part of the lit surfaces can bee seen in the rendered image.
          * In general, this type of scene is problematic for any photon tracing based methods including progressive 
          photon mapping and the original photon mapping algorithm. 
          * In this paper, we propose a simple, automatic and robust photon tracing algorithm that extends the types of 
          scenes that can be rendered efficiently with photon tracing based methods.
            * The key idea is a new importance sampling function solely based on the visibility information of each photon path.
            * In order to generate samples from this importance function, we apply two recent developments in Markov chain
            Monte Carlo methods: adaptive Markov chain sampling and replica exchange.
        * Method 
          * Overview
            * The overall idea of our algorithm is to define a visibility function of photon paths and to perform importance 
            sampling on this visibility function.
            * We define the space of this function as a hypercube.
            * We then employ local importance sampling for choosing light sources and sampling BRDFs and Russian roulette, 
            in order to generate a photon path from given random numbers.
              * In order to efficiently sample this function, we propose a combination of adaptive Markov chain sampling 
              and replica exchange.
          * Sampling Space and Visibility Function 
            * we define a photon path visibility function, $V(\vec{u})$, $\vec{u}$ a photon path in the hypercube.
              * where $V(\vec{u}) = 1$ if any photon due to this photon path contributes to the image and $V(\vec{u} = 0)$ 
              otherwise.
            * The importance function is simply the normalized version of this visibility function $V(\vec{u})$.
            * We can also easily evaluate $V\vec{u}$ by checking if a photon path splats any photon into any of measurement 
            point in the photon splatting implementation of progressive photon mapping.
          * Photon Splatting Implementation 
            * we construct an acceleration data structure of measurement points, not a photon map.
            * In other words, this algorithm is splatting photons into the measurement points, instead of gathering photons
            at each measurement point.
            * We use this splatting implementation throughout the paper, in order to immediately utilize the visibility 
            information of the current photon path to the next photon tracing.
        * _Replica exchange Monte Carlo_
          * Overview
            * The exchange Monte Carlo method is an extended ensemble Monte Carlo method where we sample Markov chains from 
            multiple distributions simultaneously.
            * The basic idea is facilitating exploration of the sampling space by bridging multiple distant peaks using 
            another smooth importance function.
              * For example, if we use a regular Markov chain Monte Carlo method to sample from an importance sampling 
              function with two peaks seperated by zeros, the Markov chain can get trapped within an one peak for many 
              interations. The replica exchange Monte Carlo meethod can avoid this problem by introducing an extra 
              Markov chain, for instance, from a uniform distribution.
            * The key idea of the replica exchange Monte Carlo method is to perform an inter-distribution exchange such that 
            the above product distribution of samples remains unchanged.
          * Our formulation 
            * the distributions with probability $r(\vec{u}_I, \vec{u}_F) = \frac{F(\vec{u}_I)I(\vec{u}_F)}{F(\vec{u}_F)I(\vec{u}_I)}$
              * $I(\vec{u}_I) = I(\vec{u}_F) = 1$
              * $r(\vec{u}_I, \vec{u}_F) = \frac{F(\vec{u}_I)1}{\frac{1}{V_c}1} = V(\vec{u}_I)$ 
          * Progressive Estimation of the Normalized Term
            * $V_c = \int V(\vec{u})d\vec{u} \approx \frac{N_{I, V(\vec{u} = 1)}}{N_{I, total}}.$
        * Adaptive Markov Chain Monte Carlo
          * Adaptive Markov chain Monte Carlo methods provide a way to automatically adjust mutation strategies during the 
          computation by learning the importance function as we sample. 
          * Since adaptive Markov chain Monte Carlo methods in general cover many different variantions, we only provide 
          an overview of the method that we use, which is a controlled Markov chain Monte Carlo method.
            * The idea of a controlled Markov chain Monte Carlo method is to adjust the parameters of given fixed mutation
            strategies based on the previous samples.
            * $\vec{\Theta}_{i+1} = \vec{\Theta}_i + H(i, \vec{Theta}_i, \vec{u}_i, ..., \vec{u}_1). Eq. (6)$ 
              * $\vec{\Theta}$ : the parameters value
              * $\vec{u}_i, ... , \vec{u}_1$ : all samples up to the i th iteration
              * $H$ is a function that computes the changes of the parameters accroding to this history of samples
              and the last parameter values $\vec{\Theta}_i$
              * One important condition that $H$ needs to satisfy in order to keep the sample distribution intact is 
              diminishing adaptation principle.
              * $\lim_{i \to \infty} H(i, \vec{Theta}_i, \vec{u}_i, ..., \vec{u}_1) = 0.$
            * one simple approach that is used in existing adaptive Markov chain Monte Carlo methods is changing the para-
            meters such as that an _acceptance ratio_ of Markov chains reaches the desired value.
              * The acceptance ratio is the fraction of accepted mutations over all the mutations.
              * We can thus simplyify Eq. (6) as :
                * $\vec{\Theta}_{i+1} = \vec{\Theta}_i + H(i, A^*, A_i). Eq. (8)$ 
                * $A^*$ is the target acceptance ratio 
                * $A_i$ is the acceptance ratio of samples up to $i$.
          * Our Formulation 
            * $\Delta u = sgn (2\xi_0 - 1) \xi_1^{\frac{1}{\Theta_i}+1}$
              * $\Theta_i$ is the adaptive mutation size at the $i$ th Markov chain
              * $sgn(x)$ is a function that returns the sign of $x$
              * $\xi_0 and \xi_1$ are uniform random number within $(0, 1)$
            * we use a simple form of a controlled Markov chain Monte Carlo method, which adjusts a single mutation parameter in a power function.
            * acceptane probability is 
              * $a(\vec{u} + \Delta \vec{u} \leftarrow \vec{u}) = \frac{F(\vec{u} + \Delta \vec{u})}{F(\vec{u})} = \frac{V(\vec{u} + \Delta \vec{u})}{V(\vec{u})} = V(\vec{u} + \Delta\vec{u}).$
            * then update $\Theta_i$ as follows:
              * $\Theta_{i+1} = \Theta_i + \gamma_i(A_i - A^*)$        
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