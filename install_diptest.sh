if [ ! -e $HOME/devel ]; then
    mkdir $HOME/devel
fi

cd $HOME/devel
pip install -e git+http://github.com/jfear/diptest.git@bf005a8662d6e866842d5c0f387a011f773c5b04#egg=diptest-master
cd -
