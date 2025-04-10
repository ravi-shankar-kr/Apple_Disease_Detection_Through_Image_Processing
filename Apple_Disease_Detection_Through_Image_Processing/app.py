# import necessary libraries
from flask import Flask, render_template, request

import  numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model 


#load model
model = load_model("model/best_model.h5", compile=False)
print('@@ model loaded')

def pred_app_dieas(apple):
    test_image = load_img(apple, target_size = (150, 150))  #load image
    print("@@ Got Image for detection")
    
    test_image = img_to_array(test_image)/255           #convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0)   #change dimention 3d to 4d
    
    result = model.predict(test_image).round(3)          #predict disease apple or not
    print('@@ Raw result = ', result, ' Predicted Index:', np.argmax(result))
    
    pred = np.argmax(result)      # get the index of max value
    
    if pred == 0:
        return "Blotch apple", 'Blotch_apple.html'      # if index 0 blotch apple
    elif pred == 1:
        return 'Healthy apple', 'Healthy_apple.html'    #if index 1 helthy apple
    elif pred == 2:
        return 'Rot apple', 'Rot_apple.html'            # if index 2 Rot apple
    else:
        return "Scab apple", 'index.html'               # id index 3 Scab apple



# ----------->> detect_apple_diseases <<---- end

# create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# get input image from clint then predict class and render respective .html page for

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return "No file part", 400  # Return error if no file uploaded
        
        file = request.files['image']
        
        if file.filename == '':
            return "No selected file", 400  # Return error if file name is empty
        
        filename = file.filename
        print("@@ Input posted =", filename)
        
        file_path = os.path.join('static/user_uploaded', filename)  # Make sure this directory exists
        file.save(file_path)

        print("@@ detecting class......")
        pred, output_page = pred_app_dieas(apple=file_path)
        
        return render_template(output_page, pred_output=pred, user_image=file_path)
    
    except Exception as e:
        print("Error:", str(e))  # Print error in the console
        return "Error occurred: " + str(e), 500  # Return error message

 


# for local system & cloud
if __name__ == "__main__":
    app.run(threaded=False)





