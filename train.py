import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

# Lets Ignore the "mps" option for the example
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TOGGLE THE ERROR BY SETTING A LARGER image_size
image_size = 400 


def load_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    # Basic Image Loader
    loader = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    image = loader(image)[:3, :, :].unsqueeze(0)
    return image.to(device, torch.float)

# Basic Model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True)
        self.model = self.model.features[:29]

    def forward(self, inp):
        features = []
        
        for layer_mum, layer in enumerate(self.model):
            inp = layer(inp)
            if str(layer_mum) in self.chosen_features:
                features.append(inp)

        return features

# Basic Loss
def calculate_content_loss(cloned_feature, content_feature):
    return torch.mean((cloned_feature - content_feature) ** 2)


def calculate_style_loss(cloned_feature, style_feature):
    batch_size, channel, height, width = cloned_feature.shape

    cloned_gram_matrix = torch.mm(cloned_feature.view(channel, height*width), cloned_feature.view(channel, height*width).t())
    style_gram_matrix = torch.mm(style_feature.view(channel, height*width), style_feature.view(channel, height*width).t())

    return torch.mean((cloned_gram_matrix-style_gram_matrix) **2 )


def calculate_total_loss(cloned_features, content_features, style_features, alpha = 1, beta = .01):
    content_loss = 0
    style_loss = 0
    for cloned_feature, content_feature, style_feature in zip(cloned_features, content_features, style_features):
        content_loss += calculate_content_loss(cloned_feature, content_feature)
        style_loss += calculate_style_loss(cloned_feature, style_feature)
    return alpha*content_loss + beta*style_loss


# Set Variables
content_image_path = "assets/cleese_on_the_beach.jpeg"
style_image_path = "assets/monty_style.jpeg"

content_image = load_image(image_path=content_image_path, image_size=image_size)
style_image = load_image(image_path=style_image_path, image_size=image_size)
noise_image = content_image.clone().to(device, torch.float)
noise_image.requires_grad_(True)

total_steps = 2000
learning_rate = .001
alpha = 1
beta = .01

optimizer = optim.Adam([noise_image], lr=learning_rate)

model = VGG().to(device).eval()

for step in range(1, total_steps+1):
    noise_features = model(noise_image)
    content_features = model(content_image)
    style_features = model(style_image)

    total_loss = calculate_total_loss(noise_features, content_features, style_features, alpha=alpha, beta=beta)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step == total_steps:
        save_image(noise_image, "generated.png")
        print("Image Saved")