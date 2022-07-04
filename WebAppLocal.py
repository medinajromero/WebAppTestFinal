#Description: This program allows powerlifting athletes to predict their future performance in the sport when wearing gear or taking it off.

#Import the libraries
import pandas as pd
from PIL import Image
import streamlit as st

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from joblib import dump, load

#Returns the new input as a df
def get_user_input():
	# nombre, initial value, final value, default value
	#Sex_x = st.sidebar.slider('Sex', 0,1, 0)

	Sex_x = 0 #default is female
	GenderButton = st.sidebar.radio(
     "What's your gender?",
     ('Female', 'Male')
    )
	if GenderButton == 'Female':
	    Sex_x = 0
	else:
	    Sex_x = 1

	Equipment_x = 0 #default is RAW
	EquipmentButtonX = st.sidebar.radio(
     "What's your equipment?",
     ('RAW', 'NON-RAW')
    )
	if EquipmentButtonX == 'RAW':
	    Equipment_x = 0
	else:
	    Equipment_x = 1

	Age_x = st.sidebar.number_input('Current Age', 16, 80, 25)
	BodyweightKg_x = st.sidebar.number_input('Current BodyWeight', 30, 300, 70)
	#aqui hay que poner una logica que indique cuales de los movimientos quiere predecir
	#simplemente podriamos poner que el valor por defecto es 0 y si no escribe nada pues devuelve 0 igualmente
	Best3SquatKg_x = st.sidebar.number_input('Current Best Squat', 20, 700, 150)
	Best3BenchKg_x = st.sidebar.number_input('Current Best Bench Press', 20, 700, 150)
	Best3DeadliftKg_x = st.sidebar.number_input('Current Best Deadlift', 20, 700, 150)

	Equipment_y = 0 #default is RAW
	EquipmentButtonY = st.sidebar.radio(
     "What will be your equipment?",
     ('RAW', 'NON-RAW')
    )
	if EquipmentButtonY == 'RAW':
	    Equipment_y = 0
	else:
	    Equipment_y = 1

	Age_y = st.sidebar.number_input('Future Age', 16, 80, 25)
	BodyweightKg_y = st.sidebar.number_input('Future BodyWeight', 30, 300, 70)
	#como mucho, dejamos hasta 2 anios
	DiffDays = st.sidebar.number_input('Days until next competition', 0,365*2, 0)

	#Store a dictionary into a variable
	user_data = {
		'Sex_x':Sex_x,
		'Equipment_x':Equipment_x,
		'Age_x':Age_x,
		'BodyweightKg_x':BodyweightKg_x,
		'Best3SquatKg_x':Best3SquatKg_x,
		'Best3BenchKg_x':Best3SquatKg_x,
		'Best3DeadliftKg_x':Best3SquatKg_x,
		'Equipment_y':Equipment_y,
		'Age_y':Age_y,
		'BodyweightKg_y':BodyweightKg_y,
		'DiffDays':DiffDays
	}

	#Transform the data into a df
	features = pd.DataFrame(user_data,index =[0])

	#converting every value to int
	#for col in features.head():
	#	features[col] = features[col].astype(str).astype(int)

	return features

#returns the predictions
def CalculatePreds(newInput, models, xScalers, yScalers):
	preds = [] 
	for indx, model in enumerate(models):
		#to remove all the columns except the one with the movement
		targetToDrop = [x for x in targets if x != targets[indx]]
		new_entradaDropped = newInput.drop(targetToDrop, axis=1)
		print(new_entradaDropped.info())

		#for col in new_entradaDropped.head():
		#	new_entradaDropped[col] = new_entradaDropped[col].astype(str).astype(int)

		X=xScalers[indx].transform(new_entradaDropped)
		Predictions=model.predict(X)

		Predictions=yScalers[indx].inverse_transform(Predictions)

		valor = Predictions[0]
		print(valor)
		preds.append(valor)

	return preds

#Create a title and subtitle
st.write("""
# Powerlifting Raw and Non-Raw Performance Predictor:
""")

#Open and display image
#maybe it wont work and I have to type the full path
image = Image.open('backgroundCrop.png')
st.image(image, caption='Raw vs Non-Raw', use_column_width=True)
st.write("""This web application is capable of predicting the performance of an athlete in squat, bench press or deadlift according to the current athlete's performance and whether or not is wearing equipment (Raw or Non-Raw). HOW TO USE: Please, select or type your metrics on the left sidebar panel and the "Future Exercise Performance" section will display your prediction""")

movs = ['SQ', 'B', 'DL']

stringXScaler = 'PredictorScalerFit'
stringYScaler = 'TargetVarScalerFit'
#Recogemos los scalers
xScalers = []
yScalers = []

for mov in movs:
	xScalers.append(load('dfScalers/'+stringXScaler+mov))
	yScalers.append(load('dfScalers/'+stringYScaler+mov))

#ya tenemos los scalers de cada uno
#xScalers[0] -> ScalerX de SQ, 1 de B y 2 de DL
#y lo mismo

models = []
stringModels = 'models/ANNTunnedFinal'
#Recogemos los modelos de cada uno
for mov in movs:
	models.append(load_model(stringModels+mov))

#ya tenemos models[0] es el de SQ, 1 el de B y 2 el de DL

#Store the users input into a variable
user_input = get_user_input()

#Set a subheader and display the users input
st.subheader('Current Performance (User Input):')
st.write(user_input)

targets = ['Best3SquatKg_x', 'Best3BenchKg_x', 'Best3DeadliftKg_x']

preds = CalculatePreds(user_input, models, xScalers, yScalers)
#convert to a dataframe to print
calculated_preds = {}

#column names:
columnsOutput = ['SquatKg', 'BenchPressKg', 'DeadliftKg']
for indx, pred in enumerate(preds):
	calculated_preds[columnsOutput[indx]] = pred

#Transform the data into a df
calculated_predsDF = pd.DataFrame(calculated_preds,index =[0])

#from float to int
calculated_predsDF = calculated_predsDF[columnsOutput].astype(int)

#Set a subheader and display the classification
st.subheader('Future Exercise Performance: ')
st.write(calculated_predsDF)


#recogemos y escribimos los describe de cada df
describes = []
movHeaders = ['Squat', 'Bench Press', 'Deadlift']

for indx, mov in enumerate(movs):
	describes.append(pd.read_csv('dfDescribe/'+mov+'_Describe.csv'))
	#subheader
	st.subheader(movHeaders[indx]+' Dataset Description:')
	st.write(describes[indx])

print("end")

