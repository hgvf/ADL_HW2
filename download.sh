mkdir MC QA

wget https://www.dropbox.com/s/sudamr4w00heyod/MultipleChoice.bin?dl=0 -O ./MC/pytorch_model.bin
wget https://www.dropbox.com/s/0vtiw4jf1gr08zt/QA.bin?dl=0 -O ./QA/pytorch_model.bin

wget https://www.dropbox.com/s/w20yymn69ywgy39/MC_config.json?dl=0 -O ./MC/config.json
wget https://www.dropbox.com/s/orz2ij34yglu0y4/MC_tokenizer_config.json?dl=0 -O ./MC/tokenizer_config.json
wget https://www.dropbox.com/s/tz6hstq9dcf16hu/MC_tokenizer.json?dl=0 -O ./MC/tokenizer.json

wget https://www.dropbox.com/s/88k1grej9bydh9c/QA_config.json?dl=0 -O ./QA/config.json
wget https://www.dropbox.com/s/942e9llaws4hjul/QA_tokenizer_config.json?dl=0 -O ./QA/tokenizer_config.json
wget https://www.dropbox.com/s/9l1c1jsl3avnfcb/QA_tokenizer.json?dl=0 -O ./QA/tokenizer.json
