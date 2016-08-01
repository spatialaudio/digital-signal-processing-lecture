latex -shell-escape systems.tex
convert -units PixelsPerInch -density 144 -resize 400 systems-1.png systems-1.png 
convert -units PixelsPerInch -density 144 -resize 400 systems-2.png ../random_signals_LTI_systems/LTI_system_td.png
convert -units PixelsPerInch -density 144 -resize 400 systems-2.png ../nonrecursive_filters/LTI_system_td.png
convert -units PixelsPerInch -density 144 -resize 400 systems-3.png ../spectral_estimation_random_signals/synthesis_model.png
convert -units PixelsPerInch -density 144 -resize 400 systems-4.png ../spectral_estimation_random_signals/analysis_model.png

