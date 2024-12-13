import cv2
import numpy as np
import os

path = os.path.join(os.path.dirname(__file__), "media/img2.jpg") #resmin yolunu almak icin cok daha kolay bir yoldur
print(path)
img = cv2.imread(path) #bgr olarak oku

#img.shape[0] = yukseklik, img.shape[1] = genislik, img.shape[2] = kanal sayisi

img_width = img.shape[1] #resmin genisligi
img_height = img.shape[0] #resmin yuksekligi


#blob formatina cevirmemiz gerekiyor yolo modeline vermek icin
#(416,416) boyutunda olmasi gerekiyor cunku yolo modeli bu boyutta egitilmis

img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416),swapRB=True,crop=False)

"""print(img_blob.shape) #blob formatinda resmin boyutu"""


labels=["Good","Bad"]


#her labela karsilik gelen renk veriyoruz
colors=["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]

#renkleri numpy arrayine ceviriyoruz ki daha sonra kullanabilelim 
colors=[np.array(color.split(",")).astype("int") for color in colors] #inte cevir
colors =np.array(colors) #5,3 tek bir arary olusturuyoruz
colors= np.tile(colors,(18,1)) #18 tane renk olusturuyoruz (eger 5 den fazla nesne varsa renklerin tekrar etmesi icin)

#yolo modelini cagiriyoruz
model_path= os.path.join(os.path.dirname(__file__), "model_and_cfg/yolov3_mask_last.weights") #modelin yolu

cfg_path= os.path.join(os.path.dirname(__file__), "model_and_cfg/yolov3_mask.cfg")

if not (os.path.exists(model_path) and os.path.exists(cfg_path)):
    print("Model dosyalari bulunamadi!")
    exit()


print(model_path)
print(cfg_path)


model= cv2.dnn.readNetFromDarknet(cfg_path,model_path) #modeli oku
layers = model.getLayerNames() #modeldeki katmanlarin isimlerini al


#0. indexten basladigi icin 1 cikartiyoruz
#yeni surumunde diziye cevirdigi icin liste haline getiriyoruz
output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers().flatten()] #cikis katmanlarini al (yani tespit yapilan katmanlar)

for layer in model.getUnconnectedOutLayers():
    print(layers[layer-1])    

model.setInput(img_blob) #modelin girisine blob formatindaki resmi veriyoruz

detection_layers= model.forward(output_layer) #cikis katmanlarini icerisindeki degerlere ulasmak


######NON Max Supression --OPERARION 1 ######


ids_list = [] #tum nesnelerin idlerini tutuyoruz
boxes_list = [] #tum nesnelerin kutularini tutuyoruz
confidences_list = [] #tum nesnelerin skorlarini tutuyoruz



#################




for detectin_layer in detection_layers: #her bir tespit katmani icin

    for object_detection in detectin_layer: #her bir tespit icin

        scores = object_detection[5:] #ilk 5 degeri almiyoruz cunku ilk 5 deger nesnenin konumunu ve boyutunu veriyor

        predicted_id = np.argmax(scores) #en yuksek skora sahip olan nesnenin index numarasini aliyoruz

        confidence = scores[predicted_id] #en yuksek skora sahip olan nesnenin skorunu aliyoruz

        if confidence>0.3:
            label = labels[predicted_id] #en yuksek skora sahip olan nesnenin etiketini aliyoruz

            #yolonun matematiksel
            bounding_box = object_detection[0:4]* np.array([img_width,img_height,img_width,img_height]) #nesnenin konumunu aliyoruz

            (box_center_x, box_center_y,box_width,box_height) =bounding_box.astype("int") #nesnenin konumunu aliyoruz

            start_x = int(box_center_x - box_width/2) #kutunun baslangic x degeri (sol ust)
            start_y = int(box_center_y - box_height/2) #kutunun baslangic y degeri (sol ust)

            ######NON Max Supression --OPERARION 2 ######

            ids_list.append(predicted_id) #nesnenin id sini listeye ekliyoruz
            confidences_list.append(float(confidence)) #nesnenin skorunu listeye ekliyoruz
            boxes_list.append([start_x,start_y,int(box_width),int(box_height)])

            ###############

######NON Max Supression --OPERARION 3 ######

max_ids=cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4) #en yuksek skora sahip nesneleri buluyoruz bir array bicimnide donduruyor

for max_id in max_ids: #her bir nesne icin (yeni surumunde array doner eski surumunde liste donuyordu)

    max_class_id=int(max_id)  #en yuksek skora sahip nesnenin id sini aliyoruz
    box = boxes_list[max_class_id] #en yuksek skora sahip nesnenin kutusunu aliyoruz

    start_x = box[0] #kutunun baslangic x degeri (sol ust
    start_y= box[1] #kutunun baslangic y degeri (sol ust)
    box_width= box[2] #kutunun genisligi
    box_height= box[3] #kutunun yuksekligi

    predicted_id=ids_list[max_class_id] #en yuksek skora sahip nesnenin id sini aliyoruz
    label=labels[predicted_id] #en yuksek skora sahip nesnenin etiketini aliyoruz
    confidence=confidences_list[max_class_id] #en yuksek skora sahip nesnenin skorunu aliyoruz
################



    end_x = start_x+box_width #kutunun bitis x degeri (sag alt)
    end_y = start_y+box_height #kutunun bitis y degeri (sag alt)

    box_color = colors[predicted_id] #nesnenin rengini aliyoruz
    box_color = [int (each) for each in box_color]

    label = "{} : {:.2f}%".format(label,confidence*100) #etiketi ve skoru yazdiriyoruz
    print("predicted object {}".format(label)) #ekrana yazdiriyoruz

    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,2) #kutu ciziyoruz
    cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,2) #etiketi yazdiriyoruz


cv2.imshow("Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()