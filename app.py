import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import requests
import wikipedia
import random
import io
import base64
import pydeck as pdk

# Load model
@st.cache_resource
def load_model():
    model = torch.load("mobilenet_v2_full.pth", map_location="cpu", weights_only=False)
    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Animal categories
animal_categories = {
    "Mammals": ['antelope', 'badger', 'bat', 'bear', 'bison', 'boar', 'cat', 'chimpanzee', 'cow', 'coyote', 'deer',
                'dog', 'dolphin', 'donkey', 'elephant', 'fox', 'goat', 'gorilla', 'hamster', 'hare', 'hedgehog',
                'hippopotamus', 'horse', 'hyena', 'kangaroo', 'koala', 'leopard', 'lion', 'mouse', 'okapi',
                'orangutan', 'otter', 'ox', 'panda', 'pig', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros',
                'sheep', 'squirrel', 'tiger', 'whale', 'wolf', 'wombat', 'zebra'],
    "Birds": ['crow', 'duck', 'eagle', 'flamingo', 'goose', 'hornbill', 'hummingbird', 'owl', 'parrot', 'pelecaniformes',
              'penguin', 'pigeon', 'sandpiper', 'sparrow', 'swan', 'turkey', 'woodpecker'],
    "Insects": ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'dragonfly', 'fly', 'grasshopper', 'ladybugs',
                'mosquito', 'moth'],
    "Reptiles & Amphibians": ['lizard', 'snake', 'turtle'],
    "Sea Creatures": ['crab', 'dolphin', 'goldfish', 'jellyfish', 'lobster', 'octopus', 'oyster', 'seahorse', 'seal',
                      'shark', 'squid', 'starfish'],
}

# Flat list of class names
class_names = sorted(set(name for cat in animal_categories.values() for name in cat))

# Get category from class name
def get_category(name):
    for category, names in animal_categories.items():
        if name in names:
            return category
    return "Unknown"

# Adjust RGB
def adjust_channel(img, channel_idx, amount):
    img_np = np.array(img).astype(np.int16)
    img_np[:, :, channel_idx] = np.clip(img_np[:, :, channel_idx] + amount, 0, 255)
    return Image.fromarray(img_np.astype(np.uint8))

# Sidebar: dark mode
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown("""<style>body {background-color: #121212; color: #f0f0f0;}</style>""", unsafe_allow_html=True)

# Sidebar: quiz mode
quiz_mode = st.sidebar.checkbox("🎯 Quiz Mode (Guess the Animal)")

# Sidebar: filter
filter_category = st.sidebar.selectbox("🗂️ Filter by Category", ["All"] + list(animal_categories.keys()))

# Main UI
st.title("🦁 Animal Classifier with MobileNetV2")
st.write("Upload an image and I’ll tell you what animal it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Or take a photo")

image_source = uploaded_file if uploaded_file else camera_image

if image_source is not None:
    original_image = Image.open(image_source).convert("RGB")
    st.image(original_image, caption="📷 Original Image", use_container_width=True)

    # 🎯 Classification
    img_tensor = transform(original_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probabilities, 0)

    if top_class.item() >= len(class_names):
        st.error("Model predicted an unknown class.")
    elif top_prob.item() < 0.6:
        st.markdown("### 🤔 Not confident. Possibly not an animal.")
    else:
        predicted_label = class_names[top_class.item()]
        predicted_category = get_category(predicted_label)

        if filter_category != "All" and predicted_category != filter_category:
            st.warning(f"🚫 This image belongs to the '{predicted_category}' category, not '{filter_category}'.")
        else:
            if quiz_mode:
                options = random.sample(class_names, 3) + [predicted_label]
                random.shuffle(options)
                guess = st.radio("🤔 What animal do you think this is?", options)
                if guess == predicted_label:
                    st.success("🎉 Correct!")
                else:
                    st.error(f"❌ Nope, it's a `{predicted_label}`.")
            else:
                st.markdown(f"### 🧠 Predicted: `{predicted_label}` ({top_prob.item()*100:.2f}%)")
                st.markdown(f"**Category:** {predicted_category}")

            # Wikipedia info
            try:
                summary = wikipedia.summary(predicted_label, sentences=3, auto_suggest=False, redirect=True)
                url = f"https://en.wikipedia.org/wiki/{predicted_label.replace(' ', '_')}"
                st.info(f"📚 **About {predicted_label}:**\n\n{summary}\n\n[🔗 Read more]({url})")
            except wikipedia.exceptions.DisambiguationError as e:
                st.warning(f"⚠️ Too many options for '{predicted_label}', e.g. {e.options[:3]}")
            except Exception as e:
                st.warning(f"Could not fetch Wikipedia info: {e}")

            # Geographic Distribution Map (mocked)
            st.markdown("## 🌍 Geographic Distribution (Example Location)")
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=10, longitude=10, zoom=1),
                layers=[
                    pdk.Layer("ScatterplotLayer",
                              data=[{"position": [10, 10], "size": 100}],
                              get_position="position",
                              get_radius="size",
                              get_fill_color=[200, 30, 0, 160],
                              pickable=True)
                ]
            ))

    # 🎨 Image filters
    st.markdown("## 🎨 Image Filters (Optional)")
    modified_image = original_image.copy()

    col1, col2 = st.columns(2)

    with col1:
        red_val = st.slider("🔴 Adjust Red", -100, 100, 0)
        green_val = st.slider("🟢 Adjust Green", -100, 100, 0)
        blue_val = st.slider("🔵 Adjust Blue", -100, 100, 0)
    with col2:
        apply_invert = st.checkbox("🎨 Invert Colors")
        apply_grayscale = st.checkbox("🖤 Grayscale")
        apply_edge = st.checkbox("✏️ Edge Detection")
        apply_cartoon = st.checkbox("🖍️ Cartoon Effect")

    if red_val != 0:
        modified_image = adjust_channel(modified_image, 0, red_val)
    if green_val != 0:
        modified_image = adjust_channel(modified_image, 1, green_val)
    if blue_val != 0:
        modified_image = adjust_channel(modified_image, 2, blue_val)
    if apply_grayscale:
        modified_image = ImageOps.grayscale(modified_image).convert("RGB")
    if apply_invert:
        modified_image = ImageOps.invert(modified_image)
    if apply_edge:
        import cv2
        img_np = np.array(modified_image)
        edges = cv2.Canny(img_np, 100, 200)
        modified_image = Image.fromarray(edges).convert("RGB")
    if apply_cartoon:
        img_np = np.array(modified_image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_np, 9, 250, 250)
        cartoon_np = cv2.bitwise_and(color, color, mask=edges)
        modified_image = Image.fromarray(cartoon_np)

    st.image(modified_image, caption="🧪 Modified Image", use_container_width=True)

    # Save/share functionality
    buf = io.BytesIO()
    modified_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("📥 Download Modified Image", byte_im, file_name="modified_image.png", mime="image/png")
