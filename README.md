<h1> Hateful Meme Recognition <img src='https://media1.giphy.com/media/7Xtdpym8IzRj0u2HXc/giphy.gif?cid=790b76117cbbcf29e1036577d30c5ca11ccc0b66aa08cc5a&rid=giphy.gif&ct=s' width="30"> </h1>
  
Hello, everyone 🤚🏻!

This is a project based on Meta AI's Hateful Meme detection challenge. <br>
The goal is to correctly classify memes that are offensive from those that are not. 

To do this, I built a neural network <img src='https://www.onlygfx.com/wp-content/uploads/2021/09/brain-clipart.png' width="20"> that can take in images, text and knowledge extracted from a meme. 

<p align="center">
  <img src="/images/ArchAltoLivello.png" />
</p>


If you want to know more, read on. 

<h2> Images 🖼 </h2>
The first step is to extract knowledge from the image.<br>
<br>
The visual and textual knowledge extraction part can be found in the tutorial file in the repo. <br>
ImageAI system to recognise objects in the image.<br>
DeepFace system to recognise faces in the image.<br>
Image Captioning system to describe what is in the image.<br>

<h2> Knowledge 🌍 </h2>
I created a system for extracting knowledge from the text of a meme using wikidata. <br>

<p align="center">
  <img src="/images/KGPipelineCatchy_auto.png" />
</p>

<h2> Graph-Based System 🚀 </h2>
The graph-based system architecture is as shown in the image above. 
The idea is to take the resulting graph from the meme, apply any graph embedding algorithm to it and return a vector of features. At that point, the feature vector of the knowledge graph is concatenated with the feature vector of the image and text embedding. 


