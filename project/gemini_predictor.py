import os
import time
import json
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Configure API key
genai.configure(api_key="AIzaSyBoA3jKfAADBS_phUeY4947KUexrohG1mQ")

PREFERRED_MODEL = "gemini-2.0-flash"
FALLBACK_MODEL = "gemini-1.5-flash"

def clean_json_response(text):
    """Remove markdown code fences and extra formatting."""
    text = text.strip()
    if text.startswith("```"):
        # Remove any ```json or ``
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
    if text.startswith("json"):
        text = text[len("json"):].strip()
    return text.strip()

def predict_handwriting_features(image_bytes):
    """Predict handwriting attributes from an image."""
    model_name = PREFERRED_MODEL

    for attempt in range(2):  # Try preferred then fallback
        try:
            model = genai.GenerativeModel(model_name)

            prompt = (
                "You are a handwriting analysis AI. "
                "From the given handwriting sample, predict:\n"
                "1. Gender of the writer\n"
                "2. Whether they are left-handed or right-handed\n"
                "3. Approximate age group (child, teenager, adult, senior)\n"
                "4. Notable style traits.\n"
                "Respond ONLY with a valid JSON object.\n"
                "Keys: gender, handedness, age_group, style_traits."
            )

            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
            )

            raw_text = response.text
            print(f"🔍 Raw Model Output:\n{raw_text}\n")

            clean_text = clean_json_response(raw_text)
            result = json.loads(clean_text)
            # Standardize keys for missing fields:
            result.setdefault("gender", "Unknown")
            result.setdefault("handedness", "Unknown")
            result.setdefault("age_group", "Unknown")
            result.setdefault("style_traits", {})

            return result

        except ResourceExhausted:
            print(f"⚠️ Quota exceeded for {model_name}. Switching to fallback...")
            model_name = FALLBACK_MODEL
            time.sleep(2)

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print("🔹 Raw text was:", raw_text)
            return {
                "gender": "Unknown",
                "handedness": "Unknown",
                "age_group": "Unknown",
                "style_traits": {}
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "gender": "Unknown",
                "handedness": "Unknown",
                "age_group": "Unknown",
                "style_traits": {}
            }

if __name__ == "__main__":
    # Change this path to your handwriting image
    image_path = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset\female\F2.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        exit()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = predict_handwriting_features(image_bytes)

    print("✅ Prediction Results:")
    print(f"Gender: {result.get('gender', 'Unknown')}")
    print(f"Handedness: {result.get('handedness', 'Unknown')}")
    print(f"Age Group: {result.get('age_group', 'Unknown')}")
    print(f"Style Traits: {result.get('style_traits', 'Unknown')}")
