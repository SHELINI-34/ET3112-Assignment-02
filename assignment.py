import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

img = cv.imread('crop_field.jpg', cv.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: crop_field.jpg not found!")
    exit()

edges = cv.Canny(img, 550, 690)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Q1a: Original Cropped Image')
axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Q1b: Canny Edge Detection')
plt.tight_layout()
plt.savefig('output_q1.png')
plt.show()
print("Q1 done")

indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

print(f"Total edge points found: {len(x)}")

plt.figure(figsize=(7, 6))
plt.scatter(x, y, s=1, c='blue', label='Edge points')
plt.gca().invert_yaxis()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Q2: Scatter Plot of Edge Points')
plt.legend()
plt.savefig('output_q2.png')
plt.show()
print("Q2 done")

coeffs = np.polyfit(x, y, 1)
m_ols = coeffs[0]
c_ols = coeffs[1]
angle_ols = abs(np.degrees(np.arctan(m_ols)))

x_line = np.linspace(x.min(), x.max(), 200)
y_ols_line = m_ols * x_line + c_ols

plt.figure(figsize=(7, 6))
plt.scatter(x, y, s=1, c='blue', label='Edge points')
plt.plot(x_line, y_ols_line, 'r-', linewidth=2,
         label=f'OLS fit  theta = {angle_ols:.2f} degrees')
plt.gca().invert_yaxis()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Q3: Ordinary Least Squares Fit')
plt.legend()
plt.savefig('output_q3.png')
plt.show()
# Q3 OLS fitting completed
print(f"Q4 - OLS angle: {angle_ols:.2f} degrees")

data = np.column_stack([x, y]).astype(float)
mean_x = np.mean(x)
mean_y = np.mean(y)
centered = data - [mean_x, mean_y]
U, S, Vt = np.linalg.svd(centered)
normal = Vt[-1]
m_tls = -normal[0] / normal[1]
c_tls = mean_y - m_tls * mean_x
angle_tls = abs(np.degrees(np.arctan(m_tls)))

y_tls_line = m_tls * x_line + c_tls
# Q4 completed


plt.figure(figsize=(7, 6))
plt.scatter(x, y, s=1, c='blue', label='Edge points')
plt.plot(x_line, y_tls_line, 'g-', linewidth=2,
         label=f'TLS fit  theta = {angle_tls:.2f} degrees')
plt.gca().invert_yaxis()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Q6: Total Least Squares Fit')
plt.legend()
plt.savefig('output_q6.png')
plt.show()
print(f"Q7 - TLS angle: {angle_tls:.2f} degrees")

ransac = RANSACRegressor(
    residual_threshold=2,
    max_trials=1000,
    random_state=42
)
ransac.fit(x.reshape(-1, 1), y)
m_ransac = ransac.estimator_.coef_[0]
c_ransac = ransac.estimator_.intercept_
angle_ransac = abs(np.degrees(np.arctan(m_ransac)))


inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask
y_ransac_line = m_ransac * x_line + c_ransac

plt.figure(figsize=(7, 6))
plt.scatter(x[inlier_mask], y[inlier_mask], s=1,
            c='blue', label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], s=1,
            c='orange', label='Outliers')
plt.plot(x_line, y_ransac_line, 'r-', linewidth=2,
         label=f'RANSAC  theta = {angle_ransac:.2f} degrees')
plt.gca().invert_yaxis()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Q10: RANSAC Fit')
plt.legend()
plt.savefig('output_q10.png')
plt.show()
print(f"Q11 - RANSAC angle: {angle_ransac:.2f} degrees")

print("=========================================")
print(f"  OLS angle    = {angle_ols:.2f} degrees")
print(f"  TLS angle    = {angle_tls:.2f} degrees")
print(f"  RANSAC angle = {angle_ransac:.2f} degrees")
print("=========================================")