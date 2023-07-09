
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
# Load input image
img = cv2.imread('sample_images/moon.tif', cv2.IMREAD_GRAYSCALE)

# Get filter size from user
s = int(input("Enter filter size (odd number): "))

# Get filter order from user
k = float(input("Enter filter order: "))


def myFilter(inputImage):
    # Create identity filter
    identity_filter = np.zeros((s, s), dtype=np.float32)
    identity_filter[int((s - 1) / 2), int((s - 1) / 2)] = 1.0
    #print(identity_filter)

    # Define standard box filter
    box_filter = np.ones((s, s), dtype=np.float32) * (1.0 / (s * s))

    # Calculate high-boost filter
    img_spatial_linear_filtered =((k + 1) * identity_filter) - (k * box_filter)
    return img_spatial_linear_filtered


# Display input and output images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Input Image')
ax[1].imshow(cv2.filter2D(img, -1, myFilter(img)), cmap='gray')
ax[1].set_title('Spatial Linear Filtered Image')
plt.show()"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QRadioButton, QSpinBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt

class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(0, 0, 1900, 950)

        self.label = QLabel(self)
        self.label.setGeometry(550, 280, 380, 280)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-weight: bold; font-size: 14px")

        self.output = QLabel(self)
        self.output.setGeometry(950, 280, 380, 280)
        self.output.setAlignment(Qt.AlignCenter)
        self.output.setStyleSheet("font-weight: bold; font-size: 14px")

        self.label_title = QLabel("Original", self)
        self.label_title.setGeometry(550, 250, 380, 30)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-weight: bold; font-size: 16px")

        self.output_title = QLabel("Filtered", self)
        self.output_title.setGeometry(950, 250, 380, 30)
        self.output_title.setAlignment(Qt.AlignCenter)
        self.output_title.setStyleSheet("font-weight: bold; font-size: 16px")

        self.button = QPushButton("Select Image", self)
        self.button.setGeometry(870, 810, 100, 30)
        self.button.clicked.connect(self.select_image)

        self.blur_label = QLabel("Blur Method", self)
        self.blur_label.setGeometry(800, 850, 100, 30)

        self.box_radio = QRadioButton("Box", self)
        self.box_radio.setGeometry(800, 870, 100, 30)
        self.box_radio.setChecked(True)  # Set the default selection to "Box"
        self.box_radio.toggled.connect(self.select_blur_method)

        self.gaussian_radio = QRadioButton("Gaussian", self)
        self.gaussian_radio.setGeometry(800, 890, 100, 30)
        self.gaussian_radio.toggled.connect(self.select_blur_method)

        self.blur_method = "box"  # Default method is box

        self.k_label = QLabel("k:", self)
        self.k_label.setGeometry(1000, 850, 30, 30)

        self.k_spinbox = QSpinBox(self)
        self.k_spinbox.setGeometry(1030, 850, 60, 30)
        self.k_spinbox.setRange(-5, 5)
        self.k_spinbox.setValue(2)
        self.k_spinbox.textChanged.connect(self.apply_filter)

        self.filter_size_label = QLabel("Filter Size:", self)
        self.filter_size_label.setGeometry(950, 890, 70, 30)

        self.filter_size_spinbox = QSpinBox(self)
        self.filter_size_spinbox.setGeometry(1030, 890, 60, 30)
        self.filter_size_spinbox.setRange(3, 15)
        self.filter_size_spinbox.setSingleStep(2)  # Set step size to 2
        self.filter_size_spinbox.setValue(5)
        self.filter_size_spinbox.textChanged.connect(self.apply_filter)

        self.selected_file = ""

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)", options=options
        )
        if file_name:
            self.display_original_image(file_name)
            self.selected_file = file_name
            self.apply_filter()

    def display_original_image(self, file_name):
        pixmap = QPixmap(file_name)
        self.label.setPixmap(pixmap.scaled(350, 250))

    def myFilter(self, inputImage, k, s):
        # Create identity filter
        identity_filter = np.zeros((s, s), dtype=np.float32)
        identity_filter[int((s - 1) / 2), int((s - 1) / 2)] = 1.0
        if self.blur_method == "box":
            # Define standard box filter
            box_filter = np.ones((s, s), dtype=np.float32) * (1.0 / (s * s))

            # Calculate high-boost filter
            img_spatial_linear_filtered = ((k + 1) * identity_filter) - (k * box_filter)

            # Apply filter to input image
            img_filtered = cv2.filter2D(inputImage, -1, img_spatial_linear_filtered)
            #print(img_spatial_linear_filtered)

        elif self.blur_method == "gaussian":

            # Apply Gaussian blur to input image
            img_blurred = cv2.GaussianBlur(inputImage, (s, s), 0)
            img_filtered = cv2.addWeighted(inputImage, 1 + k, img_blurred, -k, 0)

        return img_filtered

    def select_blur_method(self):
        if self.box_radio.isChecked():
            self.blur_method = "box"
        elif self.gaussian_radio.isChecked():
            self.blur_method = "gaussian"

        self.apply_filter()

    def apply_filter(self):
        img = cv2.imread(self.selected_file)
        k = self.k_spinbox.value()
        s = self.filter_size_spinbox.value()

        # Ensure s is an odd number
        if s % 2 == 0:
            s -= 1

        # Check if s is still even after decrementing
        if s % 2 == 0:
            s = max(s - 2, 3)

        filtered_img = self.myFilter(img, k, s)

        # Convert the filtered image to QImage
        height, width, channel = filtered_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(filtered_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Create QPixmap from QImage and display
        pixmap = QPixmap.fromImage(q_image)
        self.output.setPixmap(pixmap.scaled(350, 250))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 0, 255), 2, Qt.SolidLine))
        painter.drawRect(self.button.geometry())


if __name__ == "__main__":
    app = QApplication([])
    window = ImageWindow()
    window.show()
    app.exec_()