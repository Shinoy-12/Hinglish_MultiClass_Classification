import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from wordcloud import WordCloud
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from google.colab import files
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

## Import Dataset

dataset = pd.read_excel('MasterFile_9July_1 (Autosaved).xlsx')
dataset['Sub_Category'].value_counts()

## Label Encodding

dataset['Sub_Category'].value_counts().plot.bar(figsize=(18,5))

## Replacing transformation with true Keyword

keyword = {
    " wen " : "when ",
"thnkx " : "thank ",
"thnk " : "thank ",
" sry " : "sorry ",
" bcz " : "because ",
"yup" : "yeah",
"j&j" : "johnson",
"jhonson " : "johnson",
"jnj" : "johnson",
"johnsons" : "johnson",
"jnsn" : "johnson",
"jhnsn" : "johnson",
"jonson" : "johnson",
"j$j" : "johnson",
"j& j" : "johnson",
" gohnsons" : " johnson",
" shampo" : " shampoo",
"shampu" : "shampoo",
" sampoo " : " shampoo",
" saboon" : " soap",
" sabun " : " soap ",
"himalya " : "himalaya",
"immuno booster" : "immunobooster",
"immunoboost" : "immunobooster",
"imuno booster" : "immunobooster",
"himalay" : "himalaya",
"himalayan" : "himalaya",
"himalyan" : "himalaya",
"sabamed" : "sebamed",
"sebamed" : "sebamed",
"seba med" : "sebamed",
"aveno" : "aveeno",
"malis " : "massage ",
"malish" : "massage",
" bath " : " bathing ",
"nahlana" : "bathing",
"nahana " : "bathing ",
"nehlana" : "bathing",
"jayfal" : "nutmeg",
"jaifal" : "nutmeg",
"badam" : "almond",
"almand" : "almond",
"nehlna " : "bathing ",
"nehlake " : "bathing ",
"nehlaye " : "bathing ",
"nehlati " : "bathing ",
"nehlao " : "bathing ",
"mustella" : "mustela",
"mustalla" : "mustela",
"cetaphill" : "cetaphil",
"citafil " : "cetaphil ",
"citafile" : "cetaphil",
"citafill" : "cetaphil",
"citafils" : "cetaphil",
"citaphil " : "cetaphil ",
"citaphill" : "cetaphil",
"citafil " : "cetaphil ",
"citafill" : "cetaphil",
"momco " : "momsco ",
"exzema" : "eczema",
"rash " : "rashes ",
"raches" : "rashes",
"ghamori " : "rashes",
"ghamoriya" : "rashes",
"gamori " : "rashes ",
"gamoriya" : "rashes",
"bacteria" : "germs",
"virus" : "germs",
"phunsi" : "funsi",
"pimpls" : "pimple",
"scrabies" : "scabies",
"nvlon" : "nevlon",
"dabar" : "dabur",
"chico " : "chicco ",
"sweling" : "swelling",
" gell " : " gel ",
" jel " : " gel ",
" jell " : " gel ",
" zel " : " gel ",
" zell " : " gel ",
"boropls" : "boroplus",
"boro plus" : "boroplus",
"atola" : "atogla",
"derma dew" : "dermadew",
"soframicin" : "soframycin",
"borolene" : "boroline",
"vasiline" : "vaseline",
"vaselene" : "vaseline",
"calmine" : "calamine",
"almand" : "almond",
"fegaro" : "figaro",
"moisturizing" : "moisturizer",
"moisturising" : "moisturizer",
"moisturiser" : "moisturizer",
"deetol" : "dettol",
"detol " : "dettol ",
"sanitizer" : "sanitiser",
"snitizr" : "sanitiser",
"sanetizre" : "sanitiser",
"odmos" : "odomos",
"toothpaste" : "tooth paste",
"toothbrush" : "tooth brush",
"tisue " : "tissue ",
"pigon" : "pigeon",
"floride" : "fluoride",
"patanjali" : "patanjali",
"ptnjli" : "patanjali",
"ceralack" : "cerelac",
"ceralac" : "cerelac",
"cerelak" : "cerelac",
"nestl " : "nestle ",
"nesle" : "nestle",
"poridge" : "porridge",
"porride" : "porridge",
"porriadge" : "porridge",
"anamia" : "anemia",
"diarhea" : "diarrhea",
"diarrhoea" : "diarrhea",
"diarhoea" : "diarrhea",
"similack" : "semilac",
"similak" : "semilac",
"similac " : "semilac ",
"infamil" : "enfamil",
" nfa " : " enfamil ",
"enfaml" : "enfamil",
" nfamil" : " enfamil",
"nutrica" : "nutricia",
"nutrisia" : "nutricia",
" jagry " : " jaggery",
"babies" : "baby",
"bachhe" : "beta",
"baccho" : "beta",
"bacho " : "beta ",
"bachche" : "beta",
" mnth " : " month ",
"bacchi " : "beta ",
"parata" : "paratha",
"parantha" : "paratha",
"khichri" : "kichdi",
"kichri" : "kichdi",
"kicdi " : "kichdi ",
" hony " : " honey ",
"sahad" : "honey",
"shahad" : "honey",
" bost " : " boost ",
"pampr" : "pamper",
" pmpr " : " pamper ",
"pmprs" : "pamper",
"pamprs" : "pamper",
" huggy " : " huggies",
"huggie" : "huggies",
"hugies" : "huggies",
"allergic" : "allergy",
"alergy" : "allergy",
"gracco" : "graco",
"lactogan" : "lactogen",
"dexolac" : "dexolec",
"dexlac" : "dexolec",
"verkha" : "verka",
"medella" : "medela",
"madela" : "medela",
"madella" : "medela",
"morisson" : "morisons",
"morrison" : "morisons",
"morison" : "morisons",
"mamaearth" : "mama_earth",
"mama earth" : "mama_earth",
"head ache" : "headache",
" gee " : " ghee ",
" ghe " : " ghee ",
"prega news" : "preganews",
"bio oil" : "biooil",
" cndm " : " condom ",
"sterilisation" : "sterilization",
"vomit " : "vomiting ",
"vomits" : "vomiting",
"chawanprash" : "chyanwanprash",
"chawanpras " : "chyanwanprash ",
"chavanprash" : "chyanwanprash",
"karha" : "kadha",
" kara " : " kadha ",
"karah" : "kadha",
"septicilin" : "septilin",
"nimbo " : "nimbu ",
"lemon" : "nimbu",
"mommies" : "mom",
"momies" : "mom",
"plzzzz" : "please",
"plis " : "please ",
"plz " : "please ",
"plzz" : "please",
"okkkkk" : "okay",
"okkk " : "okay ",
"oki " : "okay ",
"hey" : "hello",
"thik " : "theek ",
"khilaye" : "khilana",
"khilaya" : "khilana",
"khilao" : "khilana",
"khilakr" : "khilana",
"khata" : "khilana",
"khane" : "khana",
"khao " : "khilana ",
"kese " : "kaise ",
" paani " : "pani",
" jada " : " jayda ",
" hing " : " heeng ",
" suji " : " sooji ",
"garam" : "garmi",
" bacha" : " beta",
"vacine" : "vaccine",
"vaccination" : "vaccine",
"ilachi" : "elaichi",
"elichi" : "elaichi",
"nebulizer" : "nebuliser",
"nebulise" : "nebuliser",
"dark circle" : "darkcircle",
"alovera" : "aloevera",
"allowera" : "aloevera",
"elovera" : "aloevera",
"allovera" : "aloevera",
"aleovera" : "aloevera",
"aelovera" : "aloevera",
"aloe vera" : "aloevera",
" apka" : " aapka",
"aapke" : "aapka",
" bete " : " beta ",
"kaise" : "kaisa",
" dove" : " baby_dove",
" dov " : " baby_dove ",
" duv " : " baby_dove ",
"baby dove" : "baby_dove",
"craddle cap" : "cradle_cap",
"cardle cap" : "cradle_cap",
"cradle cap" : "cradle_cap",
"sudo cream" : "sudocream",
"repellant" : "mosquito_repellant",
"mosquito cream" : "mosquito_repellant",
"mosquito net" : "mosquito_repellant",
"musquioto" : "mosquito_repellant",
"mosquito repellant" : "mosquito_repellant",
"lifebouy" : "lifeboy",
"gharelu nuske" : "home_remedy",
"home remedies" : "home_remedy",
"home remedy" : "home_remedy",
"ice cream" : "ice_cream",
"warm water" : "warm_water",
"warmwater" : "warm_water",
"icecream" : "ice_cream",
"cold water" : "cold_water",
"brown bread" : "brown_bread",
"fox nuts" : "foxnuts",
"fox nut" : "foxnuts",
"enfa grow" : "enfagrow",
"enfa gro " : "enfagrow ",
"enfagro" : "enfagrow",
"swarna prashan" : "swarnaprashan",
"suvarnaprashana" : "swarnaprashan",
"weight loss" : "weight_loss",
"weigth gain" : "weigth_gain",
"diet chart" : "diet_chart",
"dietchart" : "diet_chart",
"food chart" : "food_chart",
"foodchart" : "food_chart",
"protinx" : "protinex",
"prntx" : "protinex",
"dietplan" : "diet_plan",
"diet plan" : "diet_plan",
"nappy cream" : "rash_cream",
"diaper cream" : "rash_cream",
"diaper creme" : "rash_cream",
"rash cream" : "rash_cream",
"nape cream" : "rash_cream",
"nappie cream" : "rash_cream",
"bodyache" : "body ache",
"adhrak" : "ginger",
"adrak" : "ginger",
"ilachi" : "elaichi",
"elichi" : "elaichi",
"cardamom" : "elaichi",
"budapa" : "ageing",
"bhudapa" : "ageing",
"sun cream" : "sunscreen",
"sunscream" : "sunscreen",
"in grown" : "ingrown",
"jhuri " : "wrinkle",
"jhuriya" : "wrinkle",
" shishi " : " bottle ",
" bottal " : " bottle ",
" botle " : " bottle "
}

text =[]
for i in dataset.text:
    text.append(i)

for j in keyword:
    x= []
    for z in text :
        m = str(z).lower()
        if j in m:
            x.append(m.replace(j,keyword[j]))
        else:
            x.append(m)
    text=x

df = pd.DataFrame(text)
frame = [df , dataset]
df1 = pd.concat(frame , axis = 1)
df1 = df1.drop(columns = 'text')
df1.columns = ['text', 'Sub_Category']

# Removing Stop Words
nltk.download('stopwords')

## Cleaning of Data
# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords

# Remove punctuation
df1['text'] = df1['text'].str.replace(r';|\?|\,|\\t|\*|\\n|\@|\$|\&|\^|\:|\#', ' ')

from nltk.corpus import stopwords

# remove stop words from text messages

stop_words = set(stopwords.words('english'))

df1['text'] = df1['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

## Removing the words with less than 3 frequency

X = X.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

##Split the data into training and testing datasets

x_train,x_test,y_train,y_test = model_selection.train_test_split(df1['text'],df1['Sub_Category'], test_size = 0.30, random_state=1)

##Using Pipeline

from sklearn.pipeline import Pipeline
pipln = Pipeline([('vect' , CountVectorizer()),
                 ('tfidf' , TfidfTransformer()),
                   ('clf' , RandomForestClassifier(n_estimators = 100 , criterion = 'entropy' , random_state=1))])


shinoy = pipln.fit(x_train,y_train)

#Prediction

y_pred = pipln.predict(x_test)
print(classification_report(y_test,y_pred))
probability = pipln.predict_proba(x_test)
df2 = pd.DataFrame()
df2['cleanedtext']=x_test
df2['category']=y_test
df2['prediction']=y_pred
df2.to_excel("Sub_Category_Prediction.xlsx")
