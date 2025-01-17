import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump, load
import matplotlib.pyplot as plt
from datetime import datetime

def detect_temperature(test_image, model_path, visualize, timestamp):
    """
    Function to detect temperature of given image by comparing the analyzed color clusters 
    with existing clusters in CSV file.

    Parameters:
    - image: The input image for analysis
    - model_path: Path to the model file with existing data 
    - visualize: Boolean indicating whether to visualize the results

    Returns:
    - most_similar_temperature: The temperature with the most similar color clusters
    - distances: List of distances to each temperature
    """
    # 모델 로드
    if not os.path.exists(model_path):
        print("No trained model found. Train the model first.")
        return
    
    svm_model = load(model_path)
    #test_image = cv2.imread(image_path)
    
    if test_image is not None:
        balanced_test_image = white_balance(test_image)
        test_hist = extract_histogram(balanced_test_image)

        # 예측
        prediction = svm_model.predict([test_hist])[0]
        probabilities = svm_model.predict_proba([test_hist])

        # 시각화
        if visualize:
            image_vis_path = f"./result_{timestamp}.jpg"
            visualization(test_image, prediction, image_vis_path)
        
        return prediction
    else:
        print("Test image not found or invalid.")

def white_balance(img):
    """
    Apply white balance to the input image.
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    l_mean = np.mean(l)
    l = cv2.addWeighted(l, 1.5, l, 0, -l_mean * 0.5)
    balanced_img = cv2.merge((l, a, b))
    return cv2.cvtColor(balanced_img, cv2.COLOR_LAB2BGR)

# 히스토그램 추출 함수
def extract_histogram(image, bins=(8, 8, 8)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def visualization(test_image, prediction, output_path):
    """
    Visualize the test image along with its predicted class and HSV histograms, and save as a single JPG image.

    Parameters:
    - test_image: Input test image (BGR format)
    - prediction: Predicted class for the test image
    - output_path: Path to save the combined visualization image
    """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

    # Calculate histograms for H, S, V channels
    hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0, 180])
    hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Normalize histograms for better comparison
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # Create a figure for combined visualization
    plt.figure(figsize=(10, 6))

    # Display the test image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Test Image\nPredicted Class: {prediction}")
    plt.axis("off")

    # Overlayed histogram
    plt.subplot(1, 2, 2)
    plt.plot(hist_h, color='red', label='Hue')
    plt.plot(hist_s, color='green', label='Saturation')
    plt.plot(hist_v, color='blue', label='Value')
    plt.title("Overlayed HSV Histograms")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.legend()

    # Adjust layout and save the visualization
    plt.tight_layout()
    plt.savefig(output_path, format='jpg')
    plt.close()

    print(f"Visualization result: {output_path}")


def detect_qr_code(image_path, visualize, timestamp):
    """
    Function to detect QR code in an image and save the result to a file

    Parameters:
    - image_path: Path to the image to detect QR code

    Returns:
    - qr_data: QR code data (None if not detected)
    - width: QR code width (None if not detected)
    - height: QR code height (None if not detected)
    - top_left: Top-left coordinate of the QR code (None if not detected)
    - bottom_left: Bottom-left coordinate of the QR code (None if not detected)
    """

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    # Initialize QR code detector
    qr_detector = cv2.QRCodeDetector()

    # Detect and decode QR code
    data, bbox, _ = qr_detector.detectAndDecode(image)

    if bbox is not None:
        bbox = bbox[0]  # bbox is a numpy array with 4 coordinates
        top_left = (int(bbox[0][0]), int(bbox[0][1]))
        top_right = (int(bbox[1][0]), int(bbox[1][1]))
        bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
        bottom_left = (int(bbox[3][0]), int(bbox[3][1]))
        
        # Calculate the size of the QR code
        width = int(top_right[0] - top_left[0])
        height = int(bottom_left[1] - top_left[1])
        if visualize:
            # Draw the QR code location on the image
            cv2.line(image, top_left, top_right, (0, 255, 0), 20)
            cv2.line(image, top_right, bottom_right, (0, 255, 0), 20)
            cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 20)
            cv2.line(image, bottom_left, top_left, (0, 255, 0), 20)
        
            # Save the result image
            output_image_path = os.path.join(os.path.dirname('./'), f'result_{timestamp}_qr.jpg')
            cv2.imwrite(output_image_path, image)
            print(f"Detected QR Code image saved at: {output_image_path}")

        return data, width, height, top_left, bottom_left
    else:
        return None, None, None, None, None



def run(image_path, csv_path, v):
    # Command line argument parser setup
    # Set image paths
    image_path = image_path
    csv_path = csv_path
    visualize = v 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Call the QR code detection function
    qr_data, qr_width, qr_height, top_left, bottom_left = detect_qr_code(image_path, visualize, timestamp)
    if qr_data is None or len(qr_data) == 0:
        #print("1/QR data not detected(string length 0)", file=sys.stderr)
        return 1, null, null
    else:
        if visualize:
            print(f"QR Code data: {qr_data}")
            print(f"QR Code width: {qr_width}")
            print(f"QR Code height: {qr_height}")
            print(f"QR Code Top-left coordinate: {top_left}")
            print(f"QR Code Bottom-left coordinate: {bottom_left}")

                # Calculate the cropping coordinates
        x1, y1 = top_left
        x2, y2 = bottom_left
        if y1 < y2 and abs(x1-x2) < abs(y1-y2):	# Case #1: QR Code is at Bottom Right
            h = abs(y1-y2)
            x_crop = max(x1 - int(2.0 * h), 0)
            y_crop = max(y1 - int(0.8 * h), 0)		
        elif x1 > x2 and abs(x1-x2) > abs(y1-y2): # Case #2: QR Code is at Bottom Left
            h = abs(x1-x2)
            x_crop = max(x1 - int(0.8 * h), 0)
            y_crop = max(y1 - int(2.0 * h), 0)
        elif y1 > y2 and abs(x1-x2) < abs(y1-y2): # Case #3: QR Code is at Top Left
            h = abs(y1-y2)
            x_crop = max(x1 + int(0.2 * h), 0)
            y_crop = max(y1 - int(0.8 * h), 0)
        elif x1 < x2 and abs(x1-x2) > abs(y1-y2): # Case #4: QR Code is at Top Right
            h = abs(x1-x2)
            x_crop = max(x1 - int(0.8 * h), 0)
            y_crop = max(y1 + int(0.2 * h), 0)

                # Read the original image
        image = cv2.imread(image_path)
        if image is not None:
                # Crop the image
            cropped_image = image[y_crop:y_crop + int(1.7*h), x_crop:x_crop + int(1.7*h)]

                    # Analyze color clusters
            detected_temperature = detect_temperature(cropped_image, csv_path, visualize, timestamp)

                    # Print the nearest temperature and distances
                    # if visualize:
                    #     print("Distances to each temperature cluster:")
                    #     for temperature, distance in distances:
                    #         print(f"Distance from color cluster of temperature {temperature}: {distance:.2f}")
                    #     print(f"The nearest temperature for the cropped image is: {detected_temperature}")
                    # else:
            #print(f"{qr_data}/{detected_temperature}")
            return 0, qr_data, detected_temperature
        else:
            #print("3/Missing Images", file=sys.stderr)
            return 2, null, null



