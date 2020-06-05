<h1>Dependencies</h1> 
1. numpy

<h1>File Format</h1>  
Labels are given in text files for each image, respectively.  

<h2>Detections</h2>  
&lt;class&gt; &lt;confidence&gt; &lt;x1&gt; &lt;y1&gt; &lt;x2&gt; &lt;y2&gt;  

<h2>Ground Truths</h2>  
&lt;class&gt; &lt;x1&gt; &lt;y1&gt; &lt;x2&gt; &lt;y2&gt; 

<h1>Usage</h1>  
python3 map-calculator.py -d &lt;detection folder&gt; -t &lt;ground truth folder&gt; -i &lt;IOU threshold&gt; -m &lt;mode&gt;
