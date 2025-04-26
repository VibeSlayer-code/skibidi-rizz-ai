#skiddy skids sigma ai
# not made for skids
# this code is very simple dear core
# i have added comments in the code so that u can understand better
#I made this code in 30 minutes. Can u be buggy idk
# Copryright (c) 2025 VibeSlayer

import kagglehub # yeah i used kaggle not huggingsface idk why
import os
import cv2 # for looking at images
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split # sci-kit library
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


print(" Downloading dataset...") #main function loades the dataset
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print(" Dataset downloaded at:", path) # this is cool it tells u the path


test_dir = os.path.join(path, 'test') # tries to locate test directory in the dataset


real_dir = os.path.join(test_dir, 'REAL') # tries to find REAL folder
fake_dir = os.path.join(test_dir, 'FAKE') # tries to find FAKE  folder Yeah this is cool

if not os.path.exists(real_dir):
    print(f" Error: 'REAL' folder not found in {real_dir}") # error handler
    exit()
elif not os.path.exists(fake_dir):
    print(f" Error: 'FAKE' folder not found in {fake_dir}")
    exit()
else:
    print(" Paths verified: Dataset is ready.") #path verifier cool


def load_images(folder, label, max_images=40): # this function loads the images from the folder and resizes them to 64x64 pixels
    images = []
    labels = []
    files = os.listdir(folder)[:max_images]
    
    print(f"📂 Loading {label} images...") # skiddy skid
    for filename in tqdm(files, desc=f"Loading {label}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255.0  
            images.append(img.flatten())
            labels.append(label)
    return images, labels

real_images, real_labels = load_images(real_dir, label="REAL")
fake_images, fake_labels = load_images(fake_dir, label="FAKE")

X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

print(f" Dataset loaded: {X.shape[0]} images.") # this is the shape of the dataset


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) # this is the train test split function which splits the dataset into 80% training and 20% testing. SOO COOL

model = RandomForestClassifier(n_estimators=100, random_state=42)

print(" Training model...") # this is the model training function which trains the model automatically. Lol ngl this was hard  
model.fit(X_train, y_train)
print(" Model trained.")


y_pred = model.predict(X_test)  # this is the model prediction function which predicts the test data
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f" Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")  # this is the accuracy of the model. Ig its around 75%


def predict_image(image_path): # this is the image prediction function which predicts the image using the model
    print(f"\n Analyzing the image: {image_path}...")
    

    img = cv2.imread(image_path)
    if img is None:
        print(" Minor Error: Couldn't load image. Please check the path.")  #error handeling 
        return
    
   
    img_resized = cv2.resize(img, (64, 64)) # this resizes the image to 64x64 pixels for efficient processing
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0  # Normalize to [0, 1]
    img_flattened = img_normalized.flatten().reshape(1, -1)

    prediction = model.predict(img_flattened)
    predicted_label = le.inverse_transform(prediction)[0]
    
    print(f"Prediction: This artwork is **{predicted_label}**!") # this is the final prediction function which predicts the image using the model. 

while True:
    user_input = input("\n Enter the path to your image (or type 'exit' to quit): ").strip() #this funtion takes the path input
    if user_input.lower() == 'exit': # so that u dont remain in the code for ever cz i have added a loop function to test multiple images.
        print(" Goodbye!")
        break
    else:
        predict_image(user_input)

# All credits to Me VibeSlayer. If you are the core testing team . pls select me lol        
