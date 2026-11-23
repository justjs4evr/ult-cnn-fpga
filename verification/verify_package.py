"""Check final hashes, ZIP contents and report file links without modifying files."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(value for key,value in attrs if key=='href')

def main():
    hashes=json.loads((HERE/'file_hashes.json').read_text(encoding='utf-8'))
    for name,expected in hashes.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
    archive=HERE.parent/'ULT_CNN_FPGA_20261025.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,expected in hashes.items():
            member='fpga_release_20261025/'+name.replace('\\','/')
            assert hashlib.sha256(z.read(member)).hexdigest()==expected,name
        members=len(z.namelist())
    parser=Links();parser.feed((HERE/'FPGA_변환_검증결과.html').read_text(encoding='utf-8'))
    for link in parser.links:assert (HERE/link).is_file(),link
    rtl=json.loads((HERE/'rtl_verification.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(file_hashes_verified=len(hashes),zip_members=members,zip_bytes=archive.stat().st_size,
        rtl_vectors=rtl['vectors'],rtl_frames=rtl['frames'],weights_bytes=(HERE/'weights.bin').stat().st_size)))

if __name__=='__main__':main()
