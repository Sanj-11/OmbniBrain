import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="OmniBrain - Image Display",
    layout="wide"
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("OmniBrain")
st.header("Image Upload & Display")

st.write(
    "Upload an image in JPG, JPEG, or PNG format."
)

# ---------------------------------------------------------
# Image Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

# ---------------------------------------------------------
# Image Display
# ---------------------------------------------------------

if uploaded_file is None:

    st.info("Please select an image to continue.")

else:

    if uploaded_file.size == 0:

        st.error("The uploaded file is empty.")

    else:

        st.success("Image uploaded successfully!")

        # File information
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File type:** {uploaded_file.type}")
        st.write(
            f"**File size:** {uploaded_file.size / 1024:.2f} KB"
        )

        st.divider()

        # Display uploaded image
        st.subheader("Uploaded Image")

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            width=600
        )