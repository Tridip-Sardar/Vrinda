# Vrinda - Your Ultimate Plant Solution

## Introduction
<div>
<p>Empowering farmers, gardeners and plant enthusiasts to detect and prevent plant diseases.
instantly diagnose, connect with the community, and get weather-based alerts</p></div>


## Technology

<div>
<p>The website is built with:</p>
  
<ul>
<li>Node.js version 20.11.1</li>
<li>MongoDB version 8.3.2</li>
<li>Express version 4.19.2</li>
<li>TensorFlow version 5.3.3</li>
<li>Seaborn</li>
<li>ejs version 3.1.10</li>
<li>Cloudinary: used to store listing images</li>
<li>Passport: used for authentication</li>
</ul>
</div>

## Features:
<div>
 <ul>
<li> Users can upload their plant's image and our application will tell them the exact disease their plant is suffering from.</li>
<li> Our application offers a plant-related community where users can share and ask their queries related to plants.</li>
<li> Users can use our fertilizer calculator to estimate the required amount of fertilizer.</li>
<li> Our web app can provide weather details of a particular location and give precautions for optimal plant growth depending on the weather.</li>
<li> Unlike other apps our app not only provides disease detection but also provides a one-stop solution for farmers, gardeners or plant enthusiasts. </li>
</ul>

    
Feel free to explore, contribute, or use this as a reference for your own projects!
</div>

## Data base 
### User Schema:
<p>username (String)</p>
<p>email (String)</p>
<p>password (String)</p>
<p>createdAt (Date)</p>

### Post Schema:
<p>title (String)</p>
<p>image(String)</p>
<p>owner(ObjectId - a reference to the user schema)</p>
<p>comments(ObjectId - a reference to the Comment schema)</p>
<p>likes(Number)</p>
<p>usersLiked((ObjectId - a reference to the user schema))</p>
<p>createdAt (Date)</p>

### Comment Schema: 
<p>commentText(String)</p> 
<p>owner(ObjectId - a reference to the user schema) </p>
<p>createdAt (Date)</p>

