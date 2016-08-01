latex -shell-escape quantization.tex
convert -units PixelsPerInch -density 144 -resize 300 quantization-1.png ../quantization/model_quantization.png 
convert -units PixelsPerInch -density 144 -resize 400 quantization-2.png ../quantization/mid_tread_characteristic.png
convert -units PixelsPerInch -density 144 -resize 400 quantization-3.png ../quantization/mid_rise_characteristic.png
convert -units PixelsPerInch -density 144 -resize 400 quantization-4.png ../quantization/noise_shaping.png
convert -units PixelsPerInch -density 144 -resize 400 quantization-5.png ../quantization/ideal_ADC.png
convert -units PixelsPerInch -density 144 -resize 600 quantization-6.png ../quantization/oversampling.png
convert -units PixelsPerInch -density 144 -resize 800 quantization-7.png ../quantization/oversampling_anti_aliasing.png
