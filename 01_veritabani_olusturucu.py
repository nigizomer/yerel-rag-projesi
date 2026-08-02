from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

def veritabani_olustur():
    # 1. PDF Dosyasını Yükle
    pdf_path = Path("bolum-100/Yapay_Zeka_El_Kitabi_2026.pdf")
    if not pdf_path.exists():
        print("HATA: PDF dosyası bulunamadı! Adını kontrol et.")
        return

    print("1. Aşama: PDF okunuyor...")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # 2. Metni Parçalara Böl (Chunking)
    # Metni kelime kelime değil, 1000 karakterlik mantıklı paragraflara bölüyoruz
    print("2. Aşama: Metin küçük ve anlamlı parçalara bölünüyor...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Bilgi: Toplam {len(chunks)} adet metin parçası (chunk) oluşturuldu.")

    # 3. Embedding (Vektörleştirme) ve Veritabanı Kaydı
    # Yazıları yapay zekanın anlayacağı matematiksel koordinatlara çeviriyoruz
    print("3. Aşama: Vektör veritabanı oluşturuluyor... (Bu işlem ilk seferde 1-2 dakika sürebilir)")
    
    # Tamamen ücretsiz ve bilgisayarında çalışan hafif bir embedding modeli kullanıyoruz
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Veritabanını 'rag_veritabani' adında bir klasöre kalıcı olarak kaydediyoruz
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./rag_veritabani"
    )
    
    print("\nTEBRİKLER! Veritabanı başarıyla oluşturuldu ve bilgisayarına kaydedildi.")
    print("Artık bu veritabanına istediğimiz soruyu sorabiliriz!")

if __name__ == "__main__":
    veritabani_olustur()