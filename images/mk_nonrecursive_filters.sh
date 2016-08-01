latex -shell-escape non_recursive_filters.tex
convert -units PixelsPerInch -density 144 -resize 600 non_recursive_filters-1.png ../nonrecursive_filters/overlap_add.png
convert -units PixelsPerInch -density 144 -resize 600 non_recursive_filters-2.png ../nonrecursive_filters/overlap_save.png
convert -units PixelsPerInch -density 144 -resize 400 non_recursive_filters-3.png ../nonrecursive_filters/roundoff_model.png
