from pdfminer.high_level import extract_text as extract_text_pdfminer
import pandas as pd

caminho_pdf = r"C:\Users\welli\Downloads\ebook_monitoramento_e_analise_de_vibracao_v_horizontal_4f919094d1.pdf"
# Esta linha extrai o texto e o armazena em texto_completo
texto_completo = extract_text_pdfminer(caminho_pdf)
nome_arquivo_saida = "texto_extraido_do_pdf.txt"
with open(nome_arquivo_saida, 'w', encoding='utf-8') as arquivo:
    arquivo.write(texto_completo)

print(f"Texto salvo com sucesso em {nome_arquivo_saida}")