
from flask import Flask, render_template, request
from keras.models import load_model
#from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array 
import keras 
import numpy as np

app = Flask(__name__,template_folder='template')


model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(256,256))
	i = img_to_array(i)/255.0
	i = i.reshape(1, 256,256,3)
	p =model.predict(i)
	if p<=0.5:
		return 'Positive'
	else:
		return 'Negative'
	



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please go to hospital for check-up if your report is Positive..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
