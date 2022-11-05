importtime
importcv2
importsys
importyaml
importos
importglob
importtqdm

sys.path.insert(0,".")
#fromface_detection.OpenVinoimportOpenVinoModel
fromface_anti_spoofing.face_anti_spoofingimportOpenVinoModel_FAS

defload_yaml(path):
print(f"Loadingconfigsfrom{path}")
withopen(path,"r")asf:
returnyaml.safe_load(f)

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


yaml_cfgs="configs.yaml"
cfgs=load_yaml(yaml_cfgs)

FAS_xml=cfgs.get("openvino")["FAS_xml"]
FAS_bin=cfgs.get("openvino")["FAS_bin"]
FAS_thr=cfgs.get("FAS_thr")


#face_detection=OpenVinoModel("face_detection/models/320x320_25.xml",input_size=(320,320))
FAS=OpenVinoModel_FAS(FAS_xml,FAS_bin,FAS_thr,input_size=(112,112),device_name="CPU")


if__name__=='__main__':
count_frame=0
total_live_turn=0
total_spoof_turn=0
path_img_txt=open("./results.txt","w")

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
