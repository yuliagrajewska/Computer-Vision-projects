# Computer-Vision-projects
some projects implementing basic CV stuff including Harris corner detector, Hough transform for line detection, Histogram of oriented gradients, as well as every ML engineer wannabe's first DL project: image classification using CNNs (classics are classics for a reason)

# Harris Corner Detector
The **Harris corner detector** is commonly used to detect corners of an image. The idea is to locate points of interest where the surrounding neighborhood shows edges in more than one direction. A corner can be detected by looking at large variations in intensities within a small window when moving around. The change can be estimated using the following equation:

$$ E(u, v) = \sum_{x, y} w(x, y) [I(x + u, y + v) - I(x, y)]^2 $$


Where:  
- \( E \) is the difference between the original and the moved window,  
- \( u \) and \( v \) are the window’s displacement in x and y directions,  
- \( w(x, y) \) is the window at position \( (x, y) \),  
- \( I(x+u, y+v) \) is the intensity of the moved window,  
- \( I(x, y) \) is the intensity of the original image at position \( (x, y) \).  

The window function can be a rectangular window or a Gaussian window which gives weights to pixel \( (x, y) \). 

The above equation can be further approximated using Taylor expansion, giving the final formula as:

$$ E(u, v) \approx \begin{bmatrix} u \\ v \end{bmatrix}^T M \begin{bmatrix} u \\ v \end{bmatrix} $$

Where:  

$$ M = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} $$

---

### Corner Detection Process

1. **Color image to grayscale image conversion**  
2. **Spatial derivatives** (x and y directions → \( G_x \) and \( G_y \))  
3. **Compute products of derivatives** at every pixel (apply smoothing kernel if required)  
4. **Compute the sums of the products of derivatives** at each pixel  
5. Define the matrix \( M \) at each pixel:  
   $$ M = \begin{bmatrix} S_x^2 & S_{xy} \\ S_{xy} & S_y^2 \end{bmatrix} $$  
6. Compute the **response of the detector** at each pixel:  
   $$ R = \text{det}(M) - k (\text{trace}(M))^2 $$  
   Where:  
   - \( \text{det}(M) = S_x^2 S_y^2 - S_{xy}^2 \)  
   - \( \text{trace}(M) = S_x + S_y \)  
   - \( k \) is the sensitivity factor to separate corners from edges, typically a value close to zero (usually between 0.04 – 0.06).  
7. **Threshold on value of \( R \)**; compute non-max suppression. All windows with \( R \) greater than a certain value are corners. If \( R \) is a large negative number, it is likely an edge; otherwise, for flat regions, \( R = 0 \).
