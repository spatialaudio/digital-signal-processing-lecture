latex -shell-escape recursive_filters.tex
convert -units PixelsPerInch -density 144 -resize 600 recursive_filters-1.png ../recursive_filters/direct_form_i.png
convert -units PixelsPerInch -density 144 -resize 600 recursive_filters-2.png ../recursive_filters/direct_form_ii.png
convert -units PixelsPerInch -density 144 -resize 600 recursive_filters-3.png ../recursive_filters/direct_form_ii_t.png
convert -units PixelsPerInch -density 144 -resize 500 recursive_filters-4.png ../recursive_filters/coupled_form.png
