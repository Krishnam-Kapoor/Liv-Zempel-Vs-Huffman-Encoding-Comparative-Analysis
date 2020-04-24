## Please Remove all previously generated text files from the directory for accurate results
## Python 3
## Krishnam Kapoor
## 17EC10026

from sklearn.preprocessing import *
import numpy as np

class Data_Gen():

    def __init__(self, seq_len = 5, No_of_Symbols = 100000, filename = 'File_to_be_compressed.txt'):
        print("\nSequence length taken is: "+str(seq_len))
        print("Total Number of symbols in the file: "+str(No_of_Symbols))
        self.seq_len=seq_len
        self.No_of_Symbols=No_of_Symbols
        self.size_tpm=seq_len
        self.filename=filename

    
    def TPM(self):
        self.TPM = np.random.uniform(size=(self.size_tpm, self.size_tpm))
        self.TPM = normalize(self.TPM, axis=1, norm='l1')
        print("\nThe TPM is:\n")
        print(self.TPM)
        print("\n")

    def __H_inf(self):
        x = np.zeros(shape=self.size_tpm + 1)
        x[self.size_tpm] = 1
        A = np.append(np.transpose(self.TPM) - np.identity(self.size_tpm), [[1 for _ in range(self.size_tpm)]], axis=0)
        M = np.transpose(x)
        pi_infinity = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(M))
        self.H_infinity = np.sum(np.array([-x * np.log2(x) for x in pi_infinity]))

    def __Get_symbol(self, symbol_no):
        symbol_list=self.symbol_list
        row_index = self.TPM[symbol_no]
        random_no = np.random.uniform(1000) / 1000
        next_symbol = symbol_list[0]
        prob_agg = np.cumsum(row_index)
        for p in range(len(symbol_list)):
            if random_no < prob_agg[p]:
                next_symbol = symbol_list[p]
                break
        return next_symbol


    def input(self):
        self.__H_inf()
        window = self.seq_len * np.float_power(2, self.seq_len * self.H_infinity )
        self.window= np.int(np.ceil(window))
        symbol_count = len(self.TPM)
        symbol_list = ['s' + str(i) for i in range(symbol_count)]
        file = open('File_to_be_compressed.txt', 'w+')
        self.symbol_list = symbol_list
        sym_index = symbol_list[0]

        print("Generating Input File.......")

        for i in range(self.No_of_Symbols):
            file.write(sym_index + " ")
            symbol_no = int(sym_index[1:])
            sym_index = self.__Get_symbol(symbol_no)
        file.close()
        print("\nInput File Generated Successfully! Please check for File_to_be_compressed.txt in your directory")



class Compress(Data_Gen):

    def File_size(self, File):
        file_read = open(File, 'r')
        bit_count = 0
        for chunk in iter(lambda: file_read.read(3), ''):
            bit_count += len(chunk)
        file_read.close()
        return bit_count

    def Window_Dict(self, dict_name = "Binary_Dict.txt"):
        print("\nCreating dictionary for Window......\n")
        self.dict_name = dict_name
        file_read = open(self.filename, "r")
        no_of_bits = int(np.ceil(np.log2(self.seq_len)))
        bit_symbols = [bin(i)[2:].zfill(no_of_bits) for i in range(self.seq_len)]

        file_write = open(self.dict_name, "w+")
        dictionary = ''

        for chunk in iter(lambda: file_read.read(3), ''):
            symbol_no = int(chunk[1])
            dictionary = dictionary + chunk[:2]
            bit_symb = '1' + bit_symbols[symbol_no]
            file_write.write(bit_symb)
            if file_read.tell() >= self.window * 3:
                break

        file_read.close()
        file_write.close()
        self.dict=dictionary

    def Huff_Compress(self, huff_file = "Huff_compressed.txt"):

        print("\nCompressing using Huffman Encoding.....")
        number_of_symbols = self.seq_len
        symbol_suffix = list(range(self.seq_len))
        file_read = open(self.filename, "r")
        file_write = open(huff_file, "w")

        prev_symbol = 0
        for chunk in iter(lambda: file_read.read(3), ''):
            symbol_no = int(chunk[1])
            prob_row = self.TPM[prev_symbol]
            huff_input = [(x, int(prob_row[x])) for x in symbol_suffix]
            huff_codes = huffman.codebook(huff_input)
            file_write.write(huff_codes[symbol_no])
            prev_symbol = symbol_no

        file_write.close()
        file_read.close()
        print("Successfully Compressed please check the file!")

    def LZ_Compress(self, file_name = "LZ_Compressed.txt"):
        print("\nCompressing using LZ Encoding.....")
        dictionary = self.dict
        w = self.window
        file_read = open(self.filename, "r")
        file_read.seek(w * 3)

        file_write = open(file_name, "a")
        n_collections = []

        for part in iter(lambda: file_read.read(3), ''):
            string = part[:2]
            n, l = 0, -1
            for chunk in iter(lambda: file_read.read(3), ''):
                p = dictionary.find(string) / 2 + 1
                if p != 0.5:
                    l, n = int(p), n + 1
                else:
                    break
                string = string + chunk[:2]
            n_collections.append(n)

            right, left = bin(n)[2:], bin(l)[2:]
            left = left.zfill(2 * len(left) - 1)
            right = right.zfill(2 * len(right) - 1)
            file_write.write(left + right)

        file_read.close()
        file_write.close()
        print("Successfully Compressed please check the file!")

    
def main():
    Obj = Compress()
    Obj.TPM()
    Obj.input()
    Obj.Window_Dict()
    Obj.Huff_Compress()
    Obj.LZ_Compress()
    f = Obj.File_size("File_to_be_compressed.txt")
    h = Obj.File_size("Huff_compressed.txt")
    l = Obj.File_size("LZ_Compressed.txt")
    print("\nNo of bits for Trivial Encoding is: "+str(f)+" bits")
    print("No of bits for Huffman Encoding is: "+str(h)+" bits")
    print("No of bits for LZ Encoding is: "+str(l)+" bits")


def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

if __name__ == "__main__":
    install_and_import('huffman')
    main()