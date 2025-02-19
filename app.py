import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from io import BytesIO

def find_intersection(pts):
    """Calculate the intersection point of diagonals of a quadrilateral."""
    p1, p2, p3, p4 = pts

    A1 = p1[1] - p3[1]
    B1 = p3[0] - p1[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = p2[1] - p4[1]
    B2 = p4[0] - p2[0]
    C2 = A2 * p2[0] + B2 * p2[1]

    matrix = np.array([[A1, B1], [A2, B2]])
    C = np.array([C1, C2])

    return np.linalg.solve(matrix, C)

def rotate_rectangle(angle_deg, points):
    """Rotate points around the center by a given angle."""
    angle_rad = math.radians(angle_deg)
    cx, cy = find_intersection(points)

    translated_points = points - np.array([cx, cy])
    rotation_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)]
    ])

    rotated_points = np.dot(translated_points, rotation_matrix.T)
    return (rotated_points + np.array([cx, cy])).astype(np.float32)

def rotate_elevation(angle_deg, points):
    """Apply elevation rotation to points (around the X-axis)."""
    angle_rad = math.radians(angle_deg)
    cx, cy = find_intersection(points)

    translated_points = points - np.array([cx, cy])
    translated_points_3d = np.hstack((translated_points, np.zeros((translated_points.shape[0], 1))))

    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_rad), -math.sin(angle_rad)],
        [0, math.sin(angle_rad), math.cos(angle_rad)]
    ])

    rotated_points_3d = np.dot(translated_points_3d, rotation_matrix_x.T)
    return (rotated_points_3d[:, :2] + np.array([cx, cy])).astype(np.float32)

def get_bounding_box(points):
    """Calculate the bounding box of a set of points."""
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return min_x, max_x, min_y, max_y

def expand_canvas(image, new_w, new_h):
    """Expand the canvas to fit transformed image dimensions."""
    h, w = image.shape[:2]
    expanded_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 0
    tx, ty = (new_w - w) // 2, (new_h - h) // 2
    expanded_img[ty:ty + h, tx:tx + w] = image
    return expanded_img, tx, ty

def apply_transparency(image, polygon_points):
    """Apply transparency outside the specified polygon."""
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    cv.fillPoly(mask, [polygon_points], 255)

    image_bgra = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    image_bgra[mask == 0, 3] = 0
    image_bgra[mask == 255, 3] = 255
    return image_bgra

def pad_image_with_polygon(image, polygon_points, thickness):
    """Add padding to the image with a polygon extension."""
    h, w, channels = image.shape
    padded_img = np.zeros((h + thickness, w, 4), dtype=np.uint8)
    padded_img[:h, :, :] = image

    lowest_points = polygon_points[np.argsort(polygon_points[:, 1])[-3:]]
    new_points = lowest_points + np.array([0, thickness], dtype=np.int32)

    combined_polygon = np.vstack((lowest_points, new_points))
    top_points = combined_polygon[:3][np.argsort(combined_polygon[:3, 0])]
    bottom_points = combined_polygon[3:][np.argsort(combined_polygon[3:, 0])]
    middle_bottom_point=bottom_points[1]

    ordered_polygon = np.vstack((top_points, bottom_points[::-1]))
    mask = np.zeros_like(padded_img[:, :, 0], dtype=np.uint8)
    cv.fillPoly(mask, [ordered_polygon], 255)

    padded_img[mask == 255, :3] = 0
    padded_img[mask == 255, 3] = 255
    return padded_img, middle_bottom_point

def apply_one_image(img,azimuth, elevation,thickness):
    h, w = img.shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    rotated_pts = rotate_rectangle(azimuth, corners)
    elevated_pts = rotate_elevation(elevation, rotated_pts)

    min_x, max_x, min_y, max_y = get_bounding_box(elevated_pts)
    new_w = int(max_x - min_x)
    new_h = int(max_y - min_y)
    expanded_w, expanded_h = max(new_w, w), max(new_h, h)

    expanded_img, tx, ty = expand_canvas(img, expanded_w, expanded_h)
    t_corners = (corners + np.array([tx, ty])).astype(np.float32)
    t_rotated = rotate_rectangle(azimuth, t_corners)
    t_elevated = rotate_elevation(elevation, t_rotated)

    M = cv.getPerspectiveTransform(t_corners, t_elevated)
    transformed_img = cv.warpPerspective(expanded_img, M, (expanded_img.shape[1], expanded_img.shape[0]))

    tx_shift, ty_shift = -min_x - tx, -min_y - ty
    shifted_img = cv.warpAffine(transformed_img, np.float32([[1, 0, tx_shift], [0, 1, ty_shift]]), (new_w, new_h))

    transparent_img = apply_transparency(shifted_img, (t_elevated + [tx_shift, ty_shift]).astype(np.int32))
    final_img,middle_bottom_point = pad_image_with_polygon(transparent_img, (t_elevated + [tx_shift, ty_shift]).astype(np.int32), thickness=thickness)
    
    return final_img,middle_bottom_point

def apply_all_images(azimuth, elevation, thickness, order,uploaded_files):

    for uploaded_file in uploaded_files:
        if uploaded_file.name==order[0]:
            print(uploaded_file.name)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
            if img.shape[-1] == 4:  # Vérifier s'il y a un canal alpha
                img = img[:, :, :3]
            uploaded_file.seek(0)  # Reset file pointer for safety
            img_h, img_w, img_c = img.shape
            print(f"img_c={img_c}")
            assert img is not None, "File could not be read."
            final_img,middle_bottom_point = apply_one_image(img,azimuth, elevation,thickness)
            h, w, c = final_img.shape
            # Start with a transparent canvas
            empty_h=int(h*(0.5*(len(order))+0.5))
            offset_y=h/2
            empty_image = np.zeros((empty_h, w, 4), dtype=np.uint8)

    i=0
    # Load and normalize each image
    for next_image in order:
        for uploaded_file in uploaded_files:
            if uploaded_file.name==next_image:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
                if img.shape[-1] == 4:  # Vérifier s'il y a un canal alpha
                    img = img[:, :, :3]
                uploaded_file.seek(0)  # Reset file pointer for safety
                print(uploaded_file.name)
                assert img is not None, "File could not be read."
                assert img.shape == (img_h, img_w, img_c)
                final_img,middle_bottom_point = apply_one_image(img,azimuth, elevation,thickness)
                y_start = int(empty_h-h-i*offset_y)
                y_end = int(empty_h-i*offset_y)
                # Apply image to the canvas
                alpha = final_img[:, :, 3] > 0  # Non-transparent pixels
                print(f"Shape of empty_image[y_start:y_end]: {empty_image[y_start:y_end].shape}")
                print(f"Shape of alpha: {alpha.shape}")
                empty_image[y_start:y_end][alpha] = final_img[alpha]
                i+=1
    return empty_image
    

if 'permission_to_start' not in st.session_state:
    st.session_state.permission_to_start = False
if 'start_button' not in st.session_state:
    st.session_state.start_button = False

st.title('Create GIS Layers visualization')
st.write('Welcome !')
st.write('In order to start, please upload your images.')
st.write('The images should have the same resolution.')





uploaded_files = st.file_uploader(
    "Select the images (png)", 
    accept_multiple_files=True,
    type=["png"]
)

file_names=[]
if uploaded_files:
    # List to store resolutions
    resolutions = []
    for uploaded_file in uploaded_files:
        file_names.append(uploaded_file.name)
        try:
            # Convert the uploaded file to a numpy array and decode it as an image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
            if image.shape[-1] == 4:  # Vérifier s'il y a un canal alpha
                image = image[:, :, :3]
            uploaded_file.seek(0)  # Reset file pointer for safety
            # Get the resolution (height, width)
            resolution = (image.shape[1], image.shape[0],image.shape[2])  # (width, height)
            resolutions.append(resolution)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            st.session_state.permission_to_start=False
    
    # Check if all resolutions are the same
    if len(set(resolutions)) > 1:
        st.error("The uploaded images do not have the same resolution!")
        st.session_state.permission_to_start=False
        for idx, resolution in enumerate(resolutions):
            st.warning(f"Image {uploaded_files[idx].name}: Resolution = {resolution}")
    else:
        st.success(f"All images have the same resolution : {resolutions[0]}")
        st.session_state.permission_to_start=True

    order = st.multiselect(
        "In what order: Bottom to Top",
        file_names,file_names
    )
    # ordered_list=
    ordered_file_names=[]
    for i in range(len(order)):
        ordered_file_names.append(order[i])
    print(ordered_file_names)

    if st.session_state.permission_to_start:
        start_button=st.button("Start")
        if start_button:
            st.session_state.start_button = True

        
        if st.session_state.start_button:

            azimuth = st.slider("Azimuth angle", -90, 90, -45)
            elevation = st.slider("Elevation angle", 0, 90, 70)
            thickness = st.slider("Thickness", 0, 20, 5)
            result=apply_all_images(azimuth, elevation, thickness, ordered_file_names,uploaded_files)
            result_bgr = cv.cvtColor(result, cv.COLOR_RGBA2BGRA)
            st.image(result_bgr)
            # output_filename = "layers.png"
            # cv.imwrite(output_filename, result)
            # print(f"Image saved as {output_filename}")
            # Encode the image as a PNG
            _, buffer = cv.imencode(".png", result)
            # Create a BytesIO object
            result_bytes = BytesIO(buffer.tobytes())
            # Add a download button in Streamlit
            btn = st.download_button(
                label="Download image",
                data=result_bytes,
                file_name="processed_image.png",
                mime="image/png",
            )


