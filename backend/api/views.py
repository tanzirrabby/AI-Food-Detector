from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification

# Load the trained model
model_path = "photo_processor/models/dish_detection.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", num_labels=26, ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
])

# Optionally, define class names
class_names = [f"class_{i}" for i in range(26)]  # Replace with your actual class names

class ImageClassificationView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('image')
        if not file:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Open and preprocess the uploaded image
            image = Image.open(file).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # Perform classification
            with torch.no_grad():
                outputs = model(image).logits
                probs = torch.softmax(outputs, dim=1)
                
                # Get top 4 predictions
                top_probs, top_idxs = torch.topk(probs, k=4)
                top_probs = top_probs.squeeze().tolist()
                top_idxs = top_idxs.squeeze().tolist()
                
                top_predictions = [
                    {"class": class_names[idx], "confidence": round(prob, 4)}
                    for idx, prob in zip(top_idxs, top_probs)
                ]

                predicted_class = class_names[top_idxs[0]]

            return Response({
                "predicted_class": predicted_class,
                "top_4_predictions": top_predictions
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)