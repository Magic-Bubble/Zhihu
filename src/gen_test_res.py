import os
os.system("python2 main.py test --model=Boost_RNN10 --save_resmat=True --use_word=True --batch_size=64")
os.system("python2 main.py test --model=Boost_CNN10_char --save_resmat=True --use_char=True --batch_size=64")
os.system("python2 main.py test --model=Boost_CNN10 --save_resmat=True --use_word=True --batch_size=64")
os.system("python2 main.py test --model=Boost_CNN10_char_top1 --save_resmat=True --use_char=True --batch_size=64")
os.system("python2 main.py test --model=Boost_FastText10 --save_resmat=True --use_word=True --batch_size=64")
os.system("python2 main.py test --model=Boost_CNN5 --save_resmat=True --use_word=True --batch_size=64")
os.system("python2 main.py test --model=Boost_RNN1_char --save_resmat=True --use_char=True --batch_size=64")
os.system("cd utils; python2 resmat.py")