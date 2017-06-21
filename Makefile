### Directories
DATA = data
UTILS = utils
TRAIN = $(DATA)/train

LINK = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1983/ud-treebanks-v2.0.tgz?sequence=1&isAllowed=y"

### PARAMS
WINDOW =  13
BATCH = 128
EPOCHS = 20

### Dataset

$(DATA)/ud-treebanks-v2.0.tgz:
	wget $(LINK) -O $@

$(DATA)/ud-treebanks-v2.0: $(DATA)/ud-treebanks-v2.0.tgz
	cd data; tar xzvf ud-treebanks-v2.0.tgz; cd -; touch $@

$(TRAIN): $(DATA)/ud-treebanks-v2.0
	rm -rf $@;
	mkdir $@;
	for d in $</UD_*; do \
		echo $$d; \
		f=''; \
		f=`find $$d -name *train*conllu`; \
		if test $$f != ''; then \
			uddir=`echo $$d | tr '/' ' ' | rev | cut -d' ' -f1 | rev` ; \
			mkdir -p $@/$$uddir; \
			cut -f1,2 `find $$d -name *train*conllu` | $(UTILS)/converter.py > $@/$$uddir/train.tok; \
			cut -f1,2 `find $$d -name *dev*conllu` | $(UTILS)/converter.py > $@/$$uddir/dev.tok; \
		fi \
	done

$(TRAIN)/multi_train.tok: $(TRAIN)
	for t in $(TRAIN)/UD_*/*train.tok; do \
		cat $$t >> $@; \
	done

### Training


ITALIAN_TRAIN = $(TRAIN)/UD_Italian/train.tok

tokenizer_it.model: $(ITALIAN_TRAIN)
	./tokenizer.py train -epochs $(EPOCHS) -w $(WINDOW) -batch $(BATCH) -f $@ < $<
