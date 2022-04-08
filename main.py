import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import cv2 as cv
import imageio

#   ------                         PARTIE I                           -------

            # charger l'image et l'afficher sans OpenCv

print('chargement des photos')
FileName = 'pepper1.jpg'
im0 = imageio.imread(FileName) # charger l image 
plt.imshow(im0) # afficher l'image avec plt
plt.axis('off') # desactiver les axes (x,y)
plt.show()
           #charger une image avec OpenCv

im01 = cv.imread('lenna.jfif')
cv.imshow('lenna',im01) #afficher l'image avec OpenCv
cv.waitKey(0) # le programe 

          # charger une image de grand taill

          # scaling

im1 = cv.imread('pepper1.jpg')   # charger l'image
width = int(im1.shape[1]*0.75)  # on prend 75% de sa longeur original
height = int(im1.shape[0]*0.75)  # on prend 75% de sa largeur original
dimenssion = (width,height) # nv dimmenssion de l'image
im1 = cv.resize(im1,dimenssion, interpolation = cv.INTER_AREA) # cv.resize c'est la fonction qui rendre scale l'image pour bien afficher sur l'ecran
cv.imshow('pepper_resized',im1)
cv.waitKey(0)

          # sans scaling

im2 = cv.imread('pepper1.jpg',0) # charger l'image
cv.imshow('zebra',im2) #afficher l'image
cv.waitKey(0)

         # converitir l'image en niveaux de gris


im3 = 0.2125*im0[:,:,0]+ 0.7154*im0[:,:,1]+0.0721*im0[:,:,2] # multiplier les niveaux de limage avec des coef standart (détail dans le rapport) 0 :niveau de rouge 1:niveau de vert 2:niveau de bleu
plt.imshow(im3,cmap='gray',vmin=0,vmax=255) # charger l'image
plt.axis('off') # desactiver les axes
plt.show() # afficher l'image


         # codage de l'image 

  # 2 niveaux (1bit)
 
im5 = im0.copy() # faire une copie de l'image im0
nbit =1 # choisir le nombre de bits pour le codage de l'image
Ncmap = np.power(2,nbit)

delta = 255/Ncmap  # divise le domaine de l'intensité en 2^nbit
mapVal = np.zeros([Ncmap,1])
for k in range(0,Ncmap):
     mapVal[k]=(k*delta) + (delta*0.5) # des intervales de 
for k in range(0,Ncmap):
     im5[np.absolute(im5-mapVal[k])<= (0.5*delta)]=mapVal[k] # 
plt.imshow(im5,cmap='gray',vmin=0,vmax=255) # afficher l'image
plt.axis('off')
plt.show() 


   # 4 niveaux (2 bits)
im5 = im0.copy()
nbit =2
Ncmap = np.power(2,nbit)

delta = 255/Ncmap
mapVal = np.zeros([Ncmap,1])
for k in range(0,Ncmap):
     mapVal[k]=(k*delta) + (delta*0.5)
for k in range(0,Ncmap):
     im5[np.absolute(im5-mapVal[k])<= (0.5*delta)]=mapVal[k]
plt.imshow(im5,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.show()   

 # 6 niveaux
 
im5 = im0.copy()
nbit =6
Ncmap = np.power(2,nbit)

delta = 255/Ncmap
mapVal = np.zeros([Ncmap,1])
for k in range(0,Ncmap):
     mapVal[k]=(k*delta) + (delta*0.5)
for k in range(0,Ncmap):
     im5[np.absolute(im5-mapVal[k])<= (0.5*delta)]=mapVal[k]
plt.imshow(im5,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.show() 

# histogramme d'une image
 
plt.hist(im3.ravel(),255,[0,256]) # afficher l'histogramme de l'image im3 
plt.show()

plt.hist(im01.ravel(),255,[0,255]) # afficher l'histogramme de l'image im01
plt.show()


hist = cv.calcHist([im2],[0],None,[256],[0,256]) 
plt.plot(hist)

# codag de l'image 'lenna' à l'aide de l'histogramme

im5 = im01.copy()  # copie de l'image de lenna
nbit =6 # l'histogramme contient a peu pris 6 pics donc on essaie de code les images sur 6 niveaux
Ncmap = np.power(2,nbit)

delta = 255/Ncmap
mapVal = np.zeros([Ncmap,1])
for k in range(0,Ncmap):
     mapVal[k]=(k*delta) + (delta*0.5)
for k in range(0,Ncmap):
     im5[np.absolute(im5-mapVal[k])<= (0.5*delta)]=mapVal[k]
plt.imshow(im5,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.show() 

#                                     PARTIE II                                       
#   Filtrage des Images

# charger une image et convertire en niveau de girs

im01 = imageio.imread('lenna.jfif')
im6 = 0.2125*im01[:,:,0]+ 0.7154*im01[:,:,1]+0.0721*im01[:,:,2]
plt.imshow(im6,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.show()

Nx = im6.shape[0] # largeur de l'image
Ny = im6.shape[1] # longeur de l'image-
print(Nx,Ny)  # afficher les dimenssion

                     # construction de deux matrice 
    # algorithme de génére les frequences selonles deux axes 
Fx_vect = np.zeros([Nx])
Fy_vect = np.zeros([Ny])
for k in range(0,Nx):
    Fx_vect[k] = k/Nx-0.5 
for k in range(0,Ny):
    Fy_vect[k] = k/Ny-0.5
Fx_Mat = np.zeros([Nx,Ny])
Fy_Mat = np.zeros([Nx,Ny])


for k in range(0,Ny):
    Fy_Mat[0:Nx,k] = Fy_vect[k]
for k in range(0,Nx):   
    Fx_Mat[k,0:Nx] = Fx_vect[k]
    
print(Fx_Mat,Fy_Mat)


plt.imshow(im01,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.show()

dft = np.fft.fft2(im6) # on fait la transforme de fourie de l'image (spectre)

dft_cente = np.fft.fftshift(dft) # avec iffftshift on centre les petites frequences au millieu


magnitude_spectrum = 20 * np.log(np.abs(dft_cente)) # pour eviter les valeur nulles on fait le log
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8) #pour triée les valeur

plt.imshow(magnitude_spectrum,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.show()



                    #  construction de filtre passe bas 

m = int(Nx/2)
n = int(Ny/2)
r = 30 # fréquence de copure 1
#r = 70 #fréquence de copure  2
center = [m,n] # les indice de centre de l'image

for i in range(0,Nx):
    for j in range(0,Ny):
        if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= r*r :  # on calcule les case qui sont dans le cercle de rayon = fr de copure    
          Fx_Mat[i,j] = 1
          Fy_Mat[i,j] = 1
pass_bas = Fx_Mat          
  # filtrage selon l'axe X
  
plt.imshow(Fx_Mat,cmap='gray') # afficher le spectre de filtre passe bas 
plt.show()
img_filtre = Fx_Mat*dft_cente # la mulitplication de le filtre passe bas avecle spectre de l'image (filtrage fréquentille)
magnitude_spectrum = 20 * np.log(np.abs(img_filtre))
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
plt.imshow(magnitude_spectrum,cmap='gray',vmin=0,vmax=255) # afficher le resultats de multiplication
plt.axis('off')
plt.show()
idft_img_sh = np.fft.ifftshift(img_filtre) # on fait le shift inverse de spectre
img_final= np.fft.ifft2(idft_img_sh) # on fait a transforme inverse de fourier pour afficher l'image filtré
img_final=np.abs(img_final) # les valeurs devient reels
plt.imshow(img_final,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré
plt.axis('off')
plt.show()

# filtrage selon l'axe Y

plt.imshow(Fy_Mat,cmap='gray') # afficher le spectre de filtre pass bas 
plt.show()
img_filtre = Fy_Mat*dft_cente # la mulitplication de le filtre passe bas avecle spectre de l'image (filtre fréquentille)
magnitude_spectrum = 20 * np.log(np.abs(img_filtre))
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
plt.imshow(magnitude_spectrum,cmap='gray',vmin=0,vmax=255) # afficher le resultats de multiplication
plt.axis('off')
plt.show()
idft_img_sh = np.fft.ifftshift(img_filtre) # on fait le shift inverse de spectre
img_final= np.fft.ifft2(idft_img_sh) # on fait a transforme inverse de fourier pour afficher l'image filtré
img_final=np.abs(img_final) # les valeurs devient reels
plt.imshow(img_final,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré
plt.axis('off')
plt.show()




# si on filtre selonl'axe Y on choisit juste la matrice Fy_Mat 
# les resultas du filtre selon l'axe Y ou l'axe X sont les memes.

                 # construction de Filtre pass haut


 # algorithme de génére les frequences selonles deux axes
 
Fx_vect = np.zeros([Nx])
Fy_vect = np.zeros([Ny])
for k in range(0,Nx):
    Fx_vect[k] = k/Nx-0.5 
for k in range(0,Ny):
    Fy_vect[k] = k/Ny-0.5
Fx_Mat = np.zeros([Nx,Ny])
Fy_Mat = np.zeros([Nx,Ny])


for k in range(0,Ny):
    Fy_Mat[0:Nx,k] = Fy_vect[k]
for k in range(0,Nx):   
    Fx_Mat[k,0:Nx] = Fx_vect[k]



   # meme comemntaires comme le filtre pass bas
   
m = int(Nx/2)
n = int(Ny/2)
#r = 30 # fréquence de coupure 1
r = 70 # fréquence de coupure  2
center = [m,n] 

for i in range(0,Nx):
    for j in range(0,Ny):
        if (i - center[0]) ** 2 + (j - center[1]) ** 2 >= r*r :    # élimine les petite fréquences 
          Fx_Mat[i,j] = 1
          Fy_Mat[i,j] = 1

plt.imshow(Fx_Mat,cmap='gray') # afficher le filtre pass haut
plt.show()


# filtrage selon X


img_filtre = Fx_Mat*dft_cente # la mulitplication de le filtre pass haut avecle spectre de l'image (filtre fréquentille)
magnitude_spectrum = 20 * np.log(np.abs(img_filtre))
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
plt.imshow(magnitude_spectrum,cmap='gray',vmin=0,vmax=255) # afficher le resultats de multiplication
plt.axis('off')
plt.show()
idft_img_sh = np.fft.ifftshift(img_filtre) # on fait le shift inverse de spectre
img_final= np.fft.ifft2(idft_img_sh) # on fait a transforme inverse de fourier pour afficher l'image filtré
img_final=np.abs(img_final) # les valeurs devient reels
plt.imshow(img_final,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré
plt.axis('off')
plt.show()

 # filtrage selon l'axe Y
 
 
img_filtre = Fy_Mat*dft_cente # la mulitplication de le filtre pass haut avecle spectre de l'image (filtre fréquentille)
magnitude_spectrum = 20 * np.log(np.abs(img_filtre))
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
plt.imshow(magnitude_spectrum,cmap='gray',vmin=0,vmax=255) # afficher le resultats de multiplication
plt.axis('off')
plt.show()
idft_img_sh = np.fft.ifftshift(img_filtre) # on fait le shift inverse de spectre
img_final= np.fft.ifft2(idft_img_sh) # on fait a transforme inverse de fourier pour afficher l'image filtré
img_final=np.abs(img_final) # les valeurs devient reels
plt.imshow(img_final,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré
plt.axis('off')
plt.show()

# si on filtre selonl'axe Y on choisit juste la matrice Fy_Mat 
# si on veut filtrée selon les deux axes on mutiplier les matrice Fx_Mat et Fy_Mat
# les resultas du filtre selon l'axe Y ou l'axe X sont les memes.


#    le bruit blanc gaussien 

im7 = im6.copy()
gauss_bleur = cv.GaussianBlur(im7,(0,0),3,3) # on applique le bruit de gauss avec un sigma=3
plt.imshow(gauss_bleur,cmap='gray',vmin=0,vmax=255) # afficher l'image bruité
plt.axis('off')
plt.show()

# on applique le filtrage passe bas à l'image bruité (algorithme précendent)

dft = np.fft.fft2(gauss_bleur) # on fait la transforme de fourie de l'image (spectre)
dft_cente = np.fft.fftshift(dft) # avec iffftshift on centre les petites frequences au millieu
magnitude_spectrum = 20 * np.log(np.abs(dft_cente)) # pour eviter les valeur nulles on fait le log
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8) #pour triée les valeur
plt.imshow(magnitude_spectrum,cmap='gray',vmin=0,vmax=255)
plt.axis('off')
plt.show()

  
img_filtre = pass_bas*dft_cente # la mulitplication de le filtre passe bas avecle spectre de l'image (filtrage fréquentille)
magnitude_spectrum = 20 * np.log(np.abs(img_filtre))
magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
plt.imshow(magnitude_spectrum,cmap='gray',vmin=0,vmax=255) # afficher le resultats de multiplication
plt.axis('off')
plt.show()
idft_img_sh = np.fft.ifftshift(img_filtre) # on fait le shift inverse de spectre
img_final= np.fft.ifft2(idft_img_sh) # on fait a transforme inverse de fourier pour afficher l'image filtré
img_final=np.abs(img_final) # les valeurs devient reels
plt.imshow(img_final,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré
plt.axis('off')
plt.show()




     # filtre avec masque de covolution (filte moyenne)  
im8 = im01.copy()
im6 = im01.copy()
Nx = 3 # degre de filtre
Ny= 3 # degre de filtre il fault qui le soit impaire
Filtremask = np.ones([Nx,Ny]) # on cree le mask qui est une matrice contient des 1
Filtremask = Filtremask / 9 # puit on dévise sur la taille de mask

Idx = np.zeros([Nx,Ny],dtype='int8') #matrice indices lignes
Idy = np.zeros([Nx,Ny],dtype='int8') #matrice indices colonnes
for k in range(Nx):
   Idx[0:Nx,k] = k-1 #indice colonnes
   Idy[k,0:Nx] = k-1 #indice lignes
print(im6.shape[0])  # la largeur de l'image
print(im6.shape[1])  # la longeur de l'image

s = 0 # on intialise la somme des pixel adajcent de pixel traité
for i in range(0,im6.shape[0]): # pour baleyer tout les pixel d'limage
    for j in range(0,im6.shape[1]):
         s = 0
         for k in range(0,Nx):
             for p in range(0,Nx):
                 
                 ind1=np.absolute(i-Idy[k,p])  
                 ind2=np.absolute(j-Idx[k,p]) 
                 if ind1>=im6.shape[0]:
                     ind1=(im6.shape[0]-1)-(ind1-im6.shape[0]+1) # pour calculer les pixels de bordures d'images selon l'axe X 
                 if ind2 >= im6.shape[1]:
                     ind2=(im6.shape[1]-1)-(ind2-im6.shape[1]+1) # pour calculer les pixels de bordures d'images selon l'axe Y 
                
                     
                 s = s + im8[ind1,ind2] * Filtremask[k,p] # calucilent la somme des valeur des pixel (intensite ou niveau de gris) des pixels adjacent
         im8[i,j] = s # affecter ou pixel cette somme 
         
plt.imshow(im8,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré 
plt.axis('off')
plt.show()  




  
  
# ------                             PARTIE  III                      ------

img = cv.imread("grid.jpg") # images des carre noires et blanc
rows, cols, ch = img.shape

# généralement il fault 3 point pour faire une transformation géométrique affine 

cv.circle(img, (83, 90), 5, (0, 0, 255), -1) # on marque une point pour translater (petite cercle coloré en bleue)
cv.circle(img, (447, 90), 5, (0, 0, 255), -1)  
cv.circle(img, (83, 472), 5, (0, 0, 255), -1) 

pts1 = np.float32([[83, 90], [447, 90], [83, 472]]) # les trois point marqué pour translater 
pts2 = np.float32([[0, 0], [447, 90], [150, 472]])  # chenger les possition des c'est 3 point 

matrix = cv.getAffineTransform(pts1, pts2) # la fonction pour transformer les point 
result = cv.warpAffine(img, matrix, (cols, rows)) # on fait la transformation a l'image 

plt.imshow(img,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré 
plt.axis('off')
plt.show()
plt.imshow(result,cmap='gray',vmin=0,vmax=255) # afficher l'image filtré 
plt.axis('off')
plt.show()













    
         
        
 
    
 
    








