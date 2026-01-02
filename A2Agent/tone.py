
import cv2
import numpy as np

def separate_luminance(image_path, output_lum="output_luminance.jpg", output_chroma="output_chroma.jpg"):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Convert from BGR (OpenCV default) to YCrCb color space
    # Note: OpenCV uses YCrCb, where Y=Luma, Cr=Red-Difference, Cb=Blue-Difference
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 3. Split the channels
    # Y = Luminance, Cr = Chrominance Red, Cb = Chrominance Blue
    Y, Cr, Cb = cv2.split(ycrcb_img)

    # --- OUTPUT 1: LUMINANCE ONLY ---
    # The Y channel is already a grayscale image representation of luminance.
    # We save this directly.
    cv2.imwrite(output_lum, Y)
    print(f"Saved Luminance channel to: {output_lum}")

    # --- OUTPUT 2: Y FACTOR SEPARATED OUT (Chrominance Only) ---
    # To see the color without the original brightness (Y), we create a new image.
    # We cannot set Y to 0 (image would be black).
    # We set Y to a constant 128 (mid-grey) to visualize the color data purely.
    
    # Create a blank Y channel filled with 128
    const_Y = np.full_like(Y, 128)

    # Merge the constant Y with the original Cr and Cb channels
    chroma_merged = cv2.merge([const_Y, Cr, Cb])

    # Convert back to BGR so we can view/save it as a standard image
    final_chroma_img = cv2.cvtColor(chroma_merged, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(output_chroma, final_chroma_img)
    print(f"Saved Chrominance (Y-separated) image to: {output_chroma}")

    # Optional: Display the images
    cv2.imshow("Original", img)
    cv2.imshow("Luminance (Y)", Y)
    cv2.imshow("Chrominance (No Y)", final_chroma_img)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Run the function ---
# Replace 'input.jpg' with the name of your image file
if __name__ == "__main__":
    # Create a dummy image if you don't have one, or replace path below
    separate_luminance("C:\A2Agent\img.jpg")