from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2           
from tensorflow.python.keras.preprocessing import image                                            
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np                                                               
                                                                                 
model = MobileNetV2(weights='imagenet')                                    
                                                                                 
img_path = 'secadora1.jpg'                                                          
img = image.load_img(img_path, target_size=(224, 224))                           
x = image.img_to_array(img)                                                      
x = np.expand_dims(x, axis=0)                                                    
x = preprocess_input(x)                                                          
                                                                                 
preds = model.predict(x)                                                         
print ('Prediction:', decode_predictions(preds, top=1)[0][0])