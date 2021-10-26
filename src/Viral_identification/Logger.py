import csv
from os import path


class CSVLogger:
    def __init__(self,columns,file):
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()


    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header


    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self,row):
        if len(row)!=len(self.columns):
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file,"a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self
