# pip install gdown
import gdown

# = in python
# file_id_1 = "11yvzrLgIbv8e3Yy2gyCG0k2QMHepeGa0"
# gdown.download(id=file_id_1, quiet=False)
# # output = "data/dataset.zip"
# # gdown.download(id=file_id, output=output, quiet=False)
# !! this file is too big for google drive (28G)

file_id_2 = "19mmLYNT_2reMV7C7Z8pEWwgjBulVobCG"
gdown.download(id=file_id_2, quiet=False)

print()