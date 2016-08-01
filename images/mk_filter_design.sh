latex -shell-escape filter_design.tex
convert -units PixelsPerInch -density 144 -resize 600 filter_design-1.png ../filter_design/sz_mapping.png
convert -units PixelsPerInch -density 144 -resize 600 filter_design-2.png ../filter_design/RLC_lowpass.png


