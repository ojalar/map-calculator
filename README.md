<h1>Dependencies</h1> 
1. numpy

<h1>File Format</h1>  
Detections and ground truth labels are given in identically named .txt files in separate folders. Bounding box coordinates are given as upper left corner (x1, y1) and lower right corner (x2, y2).

<h2>Detections</h2>  
&lt;class&gt; &lt;confidence&gt; &lt;x1&gt; &lt;y1&gt; &lt;x2&gt; &lt;y2&gt;  

<h2>Ground Truths</h2>  
&lt;class&gt; &lt;x1&gt; &lt;y1&gt; &lt;x2&gt; &lt;y2&gt; 

<h1>Usage</h1>  
python3 map-calculator.py -d &lt;detection folder&gt; -t &lt;ground truth folder&gt; -i &lt;IoU-threshold&gt; -m &lt;mode&gt;

<h1>Modes</h1> 
Available modes are "normal", "multi" and "cluster". The "normal" and "cluster" modes are utilised in the related paper.
