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
            * $\vec{\Theta}_{i+1} = \vec{\Theta}_i + H(i, \vec{\Theta}_i, \vec{u}_i, ..., \vec{u}_1). Eq. (6)$ 
              * $\vec{\Theta}$ : the parameters value
              * $\vec{u}_i, ... , \vec{u}_1$ : all samples up to the i th iteration
              * $H$ is a function that computes the changes of the parameters accroding to this history of samples
              and the last parameter values $\vec{\Theta}_i$
              * One important condition that $H$ needs to satisfy in order to keep the sample distribution intact is 
              diminishing adaptation principle.
              * $\lim_{i \to \infty} H(i, \vec{\Theta}_i, \vec{u}_i, ..., \vec{u}_1) = 0.$
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
              * $\xi_0 \;\; and \;\; \xi_1$ are uniform random number within $(0, 1)$
            * we use a simple form of a controlled Markov chain Monte Carlo method, which adjusts a single mutation parameter in a power function.
            * acceptane probability is 
              * $a(\vec{u} + \Delta \vec{u} \leftarrow \vec{u}) = \frac{F(\vec{u} + \Delta \vec{u})}{F(\vec{u})} = \frac{V(\vec{u} + \Delta \vec{u})}{V(\vec{u})} = V(\vec{u} + \Delta\vec{u}).$
            * then update $\Theta_i$ as follows:
              * $\Theta_{i+1} = \Theta_i + \gamma_i(A_i - A^*)$        
    * **Georgiev et al. (2012), "Light transport simulation with vertex connection and merging."**
      * Takeaways
        * Although efficient global illumination algorithm exist, an acceptable approximation error in a resonable amount of time is usally only achieved for 
        specific types of input scenes. 
        * To address this problem, we present a reformulation of photon mapping as a bidirectional path sampling technique for Monte Carlo light transport simulation.
        * BPT와 PM을 결합하기 위해 $\bar{x}_s$를 추가한 것이 keypoint!
      * Paper contexts
        * Introduction
          * Efficient handling of SDS paths, on the other hand, has long been demonstrated with photon mapping (PM).
          * Indeed, PM has traditionally been combined with some of BPT's path sampling techniques through heuristic with some of BPT's path sampling techniques
          through heuristic such as global and cuastic maps, and final gathering.
            * Such a heuristic combintaiton can be far from optimal. Moreover, an adaptation of these heuristics to glossy reflectance is not obvious.
            * we can expect that a MIS-based combination of BPT and PM will yield a more robust solution than the aforementioned heuristics.
            * However, such a principled combination has not been shown so far due to important differences in the mathmatical frameworks used to formulate
            these two algorithms.
          * In this paper, we present an integration of bidirectional path tracing and photon mapping into a framework that can efficiently handle a wide range 
          of illumination effects. 
            * Our new reformulation of photon mapping as a path sampling technique allows us to employ multiple importance sampling to combine the two methods 
            in a more rubust rendering algorithm that alleviates the problem of insufficient techniques.
        * Vertex Merging 
          * Our goal is to combine photon mapping (PM) with bidirectional path tracing (BPT) into a more robust algorithm via multiple-importance sampling (MIS).
            * Scince BPT is already naturally defined in this framework (MIS-based method), we only need to reformulation the PM algorithm.
          * This involves defining the light transport paths sampled by PM with their associated pdfs that we could then plug intothe power heuristic. 
          * PM as a sampling technique for extended paths.
            * The PM radiance estimate would then then complete a full length-k extened path, which we define as $\bar{x}^* = (x_0, ... , x_s^*, x_s, ..., x_k)$
          * PM as a sampling techniques for regular paths 
            * The path $\bar{x}$ is accepted if and only if the photon location $x_s^*$ is within distance $r$ from the radiance estimate location $x_s$.
          * Path pdf
            * we first assume constant pdf $p$ inside $A_M$.
            * Second, we make the common photon mapping assumption that $A_M$ is a disk with radius $r$, and area $\pi r^2$, centered around $x_i$.
            * $p_{VM}(\bar{x}) = [\pi r^2 p (x_{s-1} \to x_s^*)] p_{VC}(\bar{x})$
        * Efficiency of Different Path Sampling Techniques 
          * Sampling densities 
          * Path reuse efficiency
    * **Mara et al. (2013), "Toward practical real-time photon mapping: Efficient GPU density estimation."**
      * Takeaways
        * We describe the design space for real-time photon density estimation, the key step of rendering global 
        illumination (GI) via photon mapping.
        * Tile based 의 performance 가 좋게 나오는 것을 확인 가능.
        * photon의 data bit size 를 표기 했음 $\to$ 추후 참고할 것
      * Paper contexts
        * Introduction
          * photon mapping contains two steps: 
            * Tracing photon along rays from light sources 
            * estimating radiance due to those photon scattering off visible surfaces (i.e. "shading").
          * Efficient parallel ray tracing hardware and software systems, such as OptiX, can trace hundreds of 
          millions of rays per second and the process can be amortized over multiple frames. 
            * Thus existing systems meet the performance needed for photon tracing.
          * In this paper, we explore the design space of architecture-aware optimization of photon shading 
          for parallel vector architectures, using a current NVIDIA GPU as a concrete, motivating case.
          * Why photon mapping ?
            * Photon mapping is a good candidate for robust, real-time rendering. 
              * It naturally simulates a range of GI phenomena.
            * A limitation of photon mapping is that does not capture transport paths that terminate in a series 
            of perfectly-specular scattering events. 
            * Those must be ray traced or approximated with screen space technique.
          * Why fast ? 
            * Offline global illumination takes minutes to hours to render with current tools.
            * Fast rendering time $\to$ It will change workflow from render-and-wait to interactive editing 
            of lighting, materials, and geometry. 
            * True real-time GI also reduces the amount of pre-baked lighting data that must be managed 
            throghout development and installed on the consumer's machine, which are significant production and 
            workflow issues today.
        * Algorithms
          * The nodes of that tree are key design choices and the leaves are eight specific algorithms. 
          * We implemented all of these, named the four best 
            * 3D Bounds
            * 2.5D Bounds 
            * HashGrid
            * Tiled 
          * Photon-Major Iteration (Scattering)
            * This implementation equation for all pixels 
              * $Eq. = L_s(X, \hat{\omega}_o) \approx \sum_{P \in \mathcal{Q}(X)} \frac{\Phi_P f_X(\hat{\omega}_P, \hat{\omega}_o) \kappa(\|Y_P - X\|)}{\int_0^{2\pi} \int_0^{r_P} \kappa(s) \, d\theta \, ds}$
              * the inner-loop's per-photon contribution is the summand in Eq.1. 
              * They rendering bounding geometry about each photon by issuing a draw call for the outer loop.
              * As with shadow volume and screen-space decal rendering, this leverages the rasterizer as a power 
              efficient generalized iterator for the inner loop. 
            * Bounding Geometry
              * 3D Polyhedral Bounds 
                * A photon volume is a polyhedron circumscribed about a photon's sphere of effect.
              * 2.5D Polygonal Bounds
                * Splatting mehtods ratserize small point primitives for photons with small radii relative to the screen resolution.
                * This allows to efficiently rasterize bounds for radii varying from very large to very small.
            * Transformation Stage
              * Vertex Stage (VS)
                * One method to render $n$ topologically identical bounding shapes is to submit $n$ instances of 
                a single bounding shape and then transform the shapes individually in the vertex stage. 
              * Geometry Stage (GS)
                * Another method for rendering $n$ bounding shapes is to submit $n$ point primitives to the GPU 
                and then amplify each into a triangle strip in the geometry stage.
          * Pixel-Major Iteration (Gathering)
            * Gather Space 
              * 3D: HashGrid 
                * A hash grid is a hash table of sparse grid cells, each of which contains an array of photons.
                * The renderer iterates over all pixels on the screen and for each one gathers photons by inde-
                xing the 3D location of a shaded point $X$ into the containing grid cell.
                * The hash grid is viewer-independent, so it can be used over multiple frames without modifica-
                tion when the lighting does change.
              * 2D: Tiles
                * A tiled algorithm inserts copies of each photons in buckets corresponding to the screen-
                space tiles it might affect. 
                * This allowed a second pass to shade allowed a second pass to shade all pixels within a tile
                from a common subset of photons that fit within shared memory for a compute shader. 
                * This yields a significant DRAM bandwidth saving.
            * Sampling Method 
              * Regular
              * Stocahstic
    * **Davidovič et al. (2014), "Progressive light transport simulation on the GPU: Survey and improvements"**
      * Survey paper!
      * Path tracing, bidirectional path tracing, photon mapping에 대한 기본적인 algorithm에 대한 설명이 있음.
    * **Evangelou et al. (2021), "Fast radius search exploiting ray tracing frameworks."**
      * Takesaways
        * HW 에서 사용가능한 Ray Tracing accelerating method 를 Radius 를 searching 하는데 응용한 기술을 보여준다.
        * 
      * Paper contents
        * Introduction 
          * all these operations about point clouds can introduce a significant computational overhead,, an issue that needs to be addressed in order to allow 
          for fast performance and scalability of the intended application.
          * Building high-quality data structures directly translates to higher query performance but generally impacts construction time negatively.
          * in this paper, we demonstrate how to leverage a highly-optimized existing ray-tracing framework in order to efficiently map the radius-search task
          to ray traversal. 
            * Central to our approach is the idea of relating the query radius with samples and, as a result, treating them as regular primitives of known bounds
            instead of simple points.
        * Radius Search using Ray Tracing 
          * A radius-search operation is defined by a set of points $S = \{s_1, s_2, ... \} \subseteq \mathbb{R}^3$ that represent the sample space 
          * A set of points $Q = \{ q_1, q_2, ...\} \subseteq \mathbb{R}^3$ that encompasses the queries to be performed.
          * $d(s,q) \le r \to I_s(q) = 1, \;\; otherwise \;\; I_s(q) = 0$ 
          * $d(s,q)$ is distance function.
            * $d(s,q) \le \tilde{r} \to I_s(q) = 1, \;\; otherwise \;\; I_s(q) = 0$ 
            * $\tilde{r} = \max_{q_i \in Q} (r_i)$
              * A subsequent rejection step is then performed $d(s_j, q_i) \le r_i$
          * Internal structure and indexing mechanics of trees built by modern BVH algorithms have some additional beneficial characteristics, aside from the 
          parallel construction.
            * First, the spatial coherence near the leaves offers infrequent tree-level changes on te loewr tree lower tree levels during traversal.
            * Second, since the input samples have relatively small bound extents, defects arising from node overlap during node splitting are less frequent.
            * Especially when the radius takes relatively small values and is progressively reduced, as is the case in progressive variants of photon mapping.
        * Radius Search via Ray Tracing 
          * First, an aixs-aligned bounding box (AABB) is constructed for every sample $s_j$  based on $\tilde{r}$ and forwarded for a regular BVH tree consturction.
          * Second, for each query $q_i$, a ray is defined with origin at $q_i$ and an infinitesimal ray extent.
          * Since sample AABBs that are potentially within range of $r_i$ must enclose it, by definition of our problem, the ray will eventually reach the leaves 
          and correctly classify potential in-radius samples $s_j$ accroding to $d(s_J, q_i)$.
    * **Kern et al. (2023), "Accelerating photon mapping for hardware-based ray tracing."**
    * **McGuire and Luebke (2009), Hardware-Accelerated Global Illumination by Image Space Photon Mapping.**
    * **Kim et al. (2019), "Caustics using screen-space photon mapping." -> Ray tracing Gems I book**
      * Takeaways
        * Photon mapping is a global illumination technique for rendering caustics and indirect lighting by simulating the transportation of photons emitted from 
        the light.
        * This chapter introduces a technique to render caustics with photon mapping in screen space with hardware ray tracing and a screen-space denoiser in 
        real time.
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