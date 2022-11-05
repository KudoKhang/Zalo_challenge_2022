importloggingaslog
fromopenvino.inference_engineimportIECore
importcv2
importnumpyasnp
importtorch
importtime
importglob
importtqdm


classOpenVinoModelFAS:
def__init__(self,model_xml,model_bin,threshold,input_size=(256,256),device_name="CPU"):
self._input_size=input_size
self.threshold=threshold
#Loadmodel
ie=IECore()
model=model_xml
log.info(f"Loadingnetwork:\n\t{model}")
net=ie.read_network(model=model,weights=model_bin)

#Defindin,out
self.input_blob=next(iter(net.input_info))

#Loadmodeltoplugin
print("Loadingmodeltotheplugin")
self.exec_net=ie.load_network(network=net,device_name=device_name)

defpredict(self,face):
start_time=time.time()
face=self._resize_and_pad(face,self._input_size)
face=self._preprocess(self,face)
res=self.exec_net.infer(inputs={self.input_blob:face,
self.input_blob:face})
score_norm=torch.softmax(torch.from_numpy(np.array(list(res.values())[3])),dim=1)[:,1]
label="Spoofing"ifscore_norm<=self.thresholdelse"Live"
print(f'{label},{score_norm.item()}.Time:{time.time()-start_time:0.4f}s')
returnlabel,score_norm.item()

@staticmethod
def_resize_and_pad(image_,size,pad_color=0):
h,w=image_.shape[:2]
sh,sw=size

#interpolationmethod
ifh>shorw>sw:#shrinkingimage
interp=cv2.INTER_AREA
else:#stretchingimage
interp=cv2.INTER_CUBIC

#aspectratioofimage
aspect=w/h#ifonPython2,youmightneedtocastasafloat:float(w)/h

#computescalingandpadsizing
ifaspect>1:#horizontalimage
new_w=sw
new_h=np.round(new_w/aspect).astype(int)
pad_vert=(sh-new_h)/2
pad_top,pad_bot=np.floor(pad_vert).astype(int),np.ceil(pad_vert).astype(int)
pad_left,pad_right=0,0

elifaspect<1:#verticalimage
new_h=sh
new_w=np.round(new_h*aspect).astype(int)
pad_horz=(sw-new_w)/2
pad_left,pad_right=np.floor(pad_horz).astype(int),np.ceil(pad_horz).astype(int)
pad_top,pad_bot=0,0

else:#squareimage
new_h,new_w=sh,sw
pad_left,pad_right,pad_top,pad_bot=0,0,0,0

iflen(image_.shape)is3andnotisinstance(pad_color,(
list,tuple,np.ndarray)):#colorimagebutonlyonecolorprovided
pad_color=[pad_color]*3

#scaleandpad
scaled_img=cv2.resize(image_,(new_w,new_h),interpolation=interp)
scaled_img=cv2.copyMakeBorder(scaled_img,pad_top,pad_bot,pad_left,pad_right,
borderType=cv2.BORDER_CONSTANT,value=pad_color)
returnscaled_img

@staticmethod
def_preprocess(self,img):
new_img=(img-127.5)/128#[-1:1]
new_img=new_img[:,:,::-1].transpose((2,0,1))
new_img=np.array(new_img)
returnnp.expand_dims(new_img,axis=0)

defcheck_path_and_result(path_txt,path_video):
name_array=[]
name_array_2=[]
path_img_txt=path_txt
withopen(path_img_txt)asf:
lines=f.readlines()
foriinlines:
path_img_txt+=1
name_array.append(i)

forvideo_nameinglob.glob(path_video+"/*"):
len_check=len(glob.glob(path_video+"/*"))
ifvideo_name.split("/")[-1]notinname_array:
name_array_2.append(video_name)

path_img_txt=open("./results_final.txt","w")
fornameinname_array:
path_img_txt.write(name)
forname_2inname_array_2:
path_img_txt.write(name_2+",0\n")

print(f"Check:{len_check}vs{len(name_array)+len(name_array_2)}")

if__name__=="__main__":
count=0
SSAN_xml="/home/thonglv/PycharmProjects/SSAN/SSAN/openvino/FAS_256s_SSAN.xml"
SSAN_bin=SSAN_xml.replace("xml","bin")
FAS_thr=0.3282
FAS=OpenVinoModelFAS(SSAN_xml,SSAN_bin,FAS_thr,input_size=(256,256))

count_frame=0
total_live_turn=0
total_spoof_turn=0
path_img_txt=open("./results_SSAN.txt","w")

root="/mnt/sdb2/data/auto_test/public_test_crop/public/videos"

forfolderintqdm.tqdm(glob.glob(root+"/*")):
video_results=[]
count_live=0
count_spoof=0
count_frame=0
path_img_txt.write(folder.split("/")[-1]+".mp4,")
path_img_txt.flush()
number=len(glob.glob(folder+"/*"))
print(folder,number)
forimginglob.glob(folder+"/*"):
#print(img)
frame=cv2.imread(img)
frame_debug=frame.copy()
t=time.time()

try:
is_spoofing,score=FAS.predict(frame_debug)
video_results.append(is_spoofing)
exceptExceptionasex:
print(ex)
continue
count_frame+=1
ifcount_frame==number:
fortype_predictinvideo_results:
iftype_predict=="Live":
count_live+=1
else:
count_spoof+=1
print(count_live,count_spoof)
ifcount_live>count_spoof:
path_img_txt.write("1\n")
path_img_txt.flush()
else:
path_img_txt.write("0\n")
path_img_txt.flush()
continue
path_img_txt.close()
check_path_and_result("./results.txt","/mnt/sdb2/data/auto_test/public_test/public/videos")
