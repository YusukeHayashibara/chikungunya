#!/usr/bin/env python3
"""
servidor.py — Dashboard Dengue/Chikungunya/Zika SP
Modelo por REGIÃO: um único SARIMA sobre a série agregada regional,
previsão escalada proporcionalmente ao município.

Instalação: pip install statsmodels numpy
Uso: python servidor.py
"""
import http.server, urllib.request, urllib.error, urllib.parse
import ssl, json, os, sys, threading, webbrowser, time, traceback
import unicodedata, concurrent.futures

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_SM = True
except ImportError:
    HAS_SM = False

HAS_SARIMA = HAS_NUMPY and HAS_SM

# ── Configuração ──────────────────────────────────────────────────────────────
PORTA = 8080
HTML  = "doencas_sp.html"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":          "application/json, */*;q=0.9",
    "Accept-Encoding": "identity",
    "Connection":      "close",
}

# ── Regiões SP ────────────────────────────────────────────────────────────────
REGIOES = {
    "São Paulo":["Arujá","Barueri","Biritiba-Mirim","Caieiras","Cajamar","Carapicuíba","Cotia","Diadema","Embu das Artes","Embu-Guaçu","Ferraz de Vasconcelos","Francisco Morato","Franco da Rocha","Guararema","Guarulhos","Itapecerica da Serra","Itapevi","Itaquaquecetuba","Jandira","Juquitiba","Mairiporã","Mauá","Mogi das Cruzes","Osasco","Pirapora do Bom Jesus","Poá","Ribeirão Pires","Rio Grande da Serra","Salesópolis","Santa Isabel","Santana de Parnaíba","Santo André","Suzano","São Bernardo do Campo","São Caetano do Sul","São Lourenço da Serra","São Paulo","Taboão da Serra","Vargem Grande Paulista","Bertioga","Cubatão","Guarujá","Itanhaém","Itariri","Mongaguá","Pedro de Toledo","Peruíbe","Praia Grande","Santos","São Vicente"],
    "Campinas":["Americana","Artur Nogueira","Campinas","Cosmópolis","Elias Fausto","Holambra","Hortolândia","Indaiatuba","Jaguariúna","Monte Mor","Nova Odessa","Paulínia","Pedreira","Santa Bárbara d'Oeste","Santo Antônio de Posse","Sumaré","Valinhos","Vinhedo","Cabreúva","Campo Limpo Paulista","Itatiba","Itupeva","Jarinu","Jundiaí","Louveira","Morungaba","Várzea Paulista","Capivari","Charqueada","Laranjal Paulista","Mombuca","Piracicaba","Rafard","Rio das Pedras","Saltinho","Santa Maria da Serra","São Pedro","Águas de São Pedro","Atibaia","Bom Jesus dos Perdões","Bragança Paulista","Joanópolis","Nazaré Paulista","Pedra Bela","Pinhalzinho","Piracaia","Socorro","Tuiuti","Vargem","Cordeirópolis","Engenheiro Coelho","Iracemápolis","Limeira","Estiva Gerbi","Itapira","Mogi Guaçu","Mogi Mirim","Aguaí","Casa Branca","Espírito Santo do Pinhal","Santa Cruz das Palmeiras","Santo Antônio do Jardim","São João da Boa Vista","Tambaú","Vargem Grande do Sul","Águas da Prata","Araras","Conchal","Leme","Santa Cruz da Conceição","Analândia","Corumbataí","Ipeúna","Rio Claro","Santa Gertrudes","Caconde","Divinolândia","Itobi","Mococa","São José do Rio Pardo","São Sebastião da Grama","Tapiratiba","Amparo","Lindóia","Monte Alegre do Sul","Serra Negra","Águas de Lindóia"],
    "Sorocaba":["Alumínio","Araçariguama","Araçoiaba da Serra","Boituva","Capela do Alto","Cerquilho","Ibiúna","Iperó","Itu","Jumirim","Mairinque","Piedade","Pilar do Sul","Porto Feliz","Salto","Salto de Pirapora","Sarapuí","Sorocaba","São Roque","Tapiraí","Tietê","Votorantim","Apiaí","Barra do Chapéu","Barão de Antonina","Bom Sucesso de Itararé","Buri","Capão Bonito","Guapiara","Itaberá","Itaóca","Itapeva","Itapirapuã Paulista","Itaporanga","Itararé","Nova Campina","Ribeira","Ribeirão Branco","Ribeirão Grande","Riversul","Taquarivaí","Barra do Turvo","Cajati","Cananéia","Eldorado","Iguape","Ilha Comprida","Iporanga","Jacupiranga","Juquiá","Miracatu","Pariquera-Açu","Registro","Sete Barras","Arandu","Avaré","Cerqueira César","Coronel Macedo","Iaras","Itaí","Manduri","Paranapanema","Taguaí","Taquarituba","Águas de Santa Bárbara","Óleo","Alambari","Angatuba","Campina do Monte Alegre","Guareí","Itapetininga","São Miguel Arcanjo","Cesário Lange","Pereiras","Porangaba","Quadra","Tatuí","Torre de Pedra"],
    "São José dos Campos":["Caçapava","Igaratá","Jacareí","Jambeiro","Monteiro Lobato","Paraibuna","Santa Branca","São José dos Campos","Campos do Jordão","Lagoinha","Natividade da Serra","Pindamonhangaba","Redenção da Serra","Santo Antônio do Pinhal","São Bento do Sapucaí","São Luís do Paraitinga","Taubaté","Tremembé","Aparecida","Canas","Cunha","Guaratinguetá","Lorena","Piquete","Potim","Roseira","Caraguatatuba","Ilhabela","São Sebastião","Ubatuba","Arapeí","Areias","Bananal","Cachoeira Paulista","Cruzeiro","Lavrinhas","Queluz","Silveiras","São José do Barreiro"],
    "Ribeirão Preto":["Altinópolis","Barrinha","Batatais","Brodowski","Cajuru","Cravinhos","Cássia dos Coqueiros","Dumont","Guariba","Guatapará","Jaboticabal","Jardinópolis","Luís Antônio","Monte Alto","Pitangueiras","Pontal","Pradópolis","Ribeirão Preto","Santa Cruz da Esperança","Santa Ernestina","Santa Rosa de Viterbo","Santo Antônio da Alegria","Serra Azul","Serrana","Sertãozinho","São Simão","Barretos","Bebedouro","Cajobi","Colina","Colômbia","Guaraci","Guaíra","Jaborandi","Monte Azul Paulista","Olímpia","Severínia","Taiaçu","Taiúva","Taquaral","Terra Roxa","Viradouro","Cristais Paulista","Franca","Itirapuã","Jeriquara","Patrocínio Paulista","Pedregulho","Restinga","Ribeirão Corrente","Rifaina","São José da Bela Vista","Ipuã","Morro Agudo","Nuporanga","Orlândia","Sales Oliveira","São Joaquim da Barra","Aramina","Buritizal","Guará","Igarapava","Ituverava","Miguelópolis"],
    "Bauru":["Agudos","Arealva","Avaí","Balbinos","Bauru","Borebi","Cabrália Paulista","Duartina","Iacanga","Lençóis Paulista","Lucianópolis","Macatuba","Paulistânia","Pederneiras","Pirajuí","Piratininga","Presidente Alves","Reginópolis","Ubirajara","Bariri","Barra Bonita","Bocaina","Boracéia","Brotas","Dois Córregos","Igaraçu do Tietê","Itaju","Itapuí","Jaú","Mineiros do Tietê","Torrinha","Anhembi","Areiópolis","Bofete","Botucatu","Conchas","Itatinga","Pardinho","Pratânia","São Manuel","Cafelândia","Guaiçara","Guarantã","Lins","Pongaí","Promissão","Sabino","Uru"],
    "Araraquara":["Américo Brasiliense","Araraquara","Boa Esperança do Sul","Borborema","Cândido Rodrigues","Dobrada","Gavião Peixoto","Ibitinga","Itápolis","Matão","Motuca","Nova Europa","Rincão","Santa Lúcia","Tabatinga","Taquaritinga","Trabiju","Descalvado","Dourado","Ibaté","Itirapina","Pirassununga","Porto Ferreira","Ribeirão Bonito","Santa Rita do Passa Quatro","São Carlos"],
    "São José do Rio Preto":["Adolfo","Altair","Bady Bassitt","Bálsamo","Cedral","Guapiaçu","Ibirá","Icém","Ipiguá","Irapuã","Jaci","José Bonifácio","Macaubal","Mendonça","Mirassol","Mirassolândia","Monte Aprazível","Neves Paulista","Nipoã","Nova Aliança","Nova Granada","Novo Horizonte","Onda Verde","Orindiúva","Palestina","Paulo de Faria","Planalto","Poloni","Potirendaba","Sales","São José do Rio Preto","Tanabi","Ubarana","Uchoa","União Paulista","Urupês","Ariranha","Catanduva","Catiguá","Elisiário","Embaúba","Fernando Prestes","Itajobi","Marapoama","Novais","Palmares Paulista","Paraíso","Pindorama","Pirangi","Santa Adélia","Tabapuã","Vista Alegre do Alto","Aparecida d'Oeste","Aspásia","Dirce Reis","Dolcinópolis","Jales","Marinópolis","Mesópolis","Palmeira d'Oeste","Paranapuã","Pontalinda","Populina","Santa Albertina","Santa Salete","Suzanápolis","São Francisco","Turmalina","Urânia","Vitória Brasil","Américo de Campos","Cardoso","Cosmorama","Floreal","Nhandeara","Parisi","Pontes Gestal","Riolândia","Sebastianópolis do Sul","Valentim Gentil","Votuporanga","Álvares Florence","Estrela d'Oeste","Fernandópolis","Guarani d'Oeste","Indiaporã","Macedônia","Meridiano","Mira Estrela","Ouroeste","Pedranópolis","São João das Duas Pontes","São João de Iracema","Nova Canaã Paulista","Rubinéia","Santa Clara d'Oeste","Santa Fé do Sul","Santa Rita d'Oeste","Santana da Ponte Pensa","Três Fronteiras"],
    "Araçatuba":["Araçatuba","Auriflama","Bento de Abreu","Gastão Vidigal","General Salgado","Guararapes","Guzolândia","Magda","Monções","Nova Castilho","Nova Luzitânia","Rubiácea","Santo Antônio do Aracanguá","Valparaíso","Alto Alegre","Avanhandava","Barbosa","Bilac","Birigui","Braúna","Brejo Alegre","Buritama","Clementina","Coroados","Gabriel Monteiro","Glicério","Lourdes","Luiziânia","Penápolis","Piacatu","Santópolis do Aguapeí","Turiúba","Zacarias","Andradina","Castilho","Guaraçaí","Ilha Solteira","Itapura","Lavínia","Mirandópolis","Murutinga do Sul","Nova Independência","Pereira Barreto","Sud Mennucci"],
    "Presidente Prudente":["Alfredo Marcondes","Anhumas","Caiabu","Emilianópolis","Estrela do Norte","Euclides da Cunha Paulista","Iepê","Indiana","João Ramalho","Martinópolis","Mirante do Paranapanema","Nantes","Narandiba","Pirapozinho","Presidente Bernardes","Presidente Prudente","Quatá","Rancharia","Regente Feijó","Ribeirão dos Índios","Rosana","Sandovalina","Santo Anastácio","Santo Expedito","Taciba","Tarabai","Teodoro Sampaio","Álvares Machado","Dracena","Flora Rica","Irapuru","Junqueirópolis","Monte Castelo","Nova Guataporanga","Ouro Verde","Panorama","Paulicéia","Santa Mercedes","São João do Pau d'Alho","Tupi Paulista","Adamantina","Flórida Paulista","Inúbia Paulista","Lucélia","Mariápolis","Osvaldo Cruz","Pacaembu","Pracinha","Sagres","Salmourão","Caiuá","Marabá Paulista","Piquerobi","Presidente Epitácio","Presidente Venceslau"],
    "Marília":["Alvinlândia","Campos Novos Paulista","Echaporã","Fernão","Garça","Getulina","Guaimbê","Gália","Júlio Mesquita","Lupércio","Marília","Ocauçu","Oriente","Oscar Bressane","Pompéia","Quintana","Vera Cruz","Álvaro de Carvalho","Assis","Borá","Cruzália","Cândido Mota","Florínia","Lutécia","Maracaí","Palmital","Paraguaçu Paulista","Pedrinhas Paulista","Platina","Tarumã","Bernardino de Campos","Canitar","Chavantes","Espírito Santo do Turvo","Ibirarema","Ipaussu","Ourinhos","Ribeirão do Sul","Salto Grande","Santa Cruz do Rio Pardo","São Pedro do Turvo","Arco-Íris","Bastos","Herculândia","Iacri","Parapuã","Queiroz","Rinópolis","Tupã","Fartura","Piraju","Sarutaiá","Tejupá","Timburi"],
}

_lock           = threading.Lock()
_muni_norm      = None   # {norm_name: geocode}
_regional_data  = {}     # (region,disease) → agg
_regional_model = {}     # (region,disease,H) → model
_city_fc_cache  = {}     # (geocode,disease,H) → payload

def _norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower())
                   if unicodedata.category(c) != 'Mn').strip()

def _norm2(s):
    """More aggressive: remove spaces and punctuation too."""
    return ''.join(c for c in _norm(s) if c.isalnum())

def _decode_response(r):
    """Lê resposta HTTP, descomprimindo gzip se necessário."""
    import gzip as _gzip
    raw = r.read()
    enc = r.headers.get("Content-Encoding", "")
    if enc == "gzip" or (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
        try:
            raw = _gzip.decompress(raw)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")

def get_muni_norm():
    global _muni_norm
    with _lock:
        if _muni_norm is not None:
            return _muni_norm
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/35/municipios"
    # Força UTF-8 sem compressão
    hdrs = {**HEADERS, "Accept-Encoding": "gzip, deflate", "Accept-Charset": "utf-8"}
    req  = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
        text = _decode_response(r)
    data     = json.loads(text)
    mapping  = {_norm(m["nome"]): str(m["id"]) for m in data}
    mapping2 = {_norm2(m["nome"]): str(m["id"]) for m in data}
    with _lock:
        _muni_norm = (mapping, mapping2)
    return _muni_norm

def geocode_of_name(name):
    m1, m2 = get_muni_norm()
    return m1.get(_norm(name)) or m2.get(_norm2(name))

def region_of_geocode(geocode):
    m1, _ = get_muni_norm()
    rev = {v: k for k, v in m1.items()}
    nome_norm = rev.get(str(geocode), "")
    for reg, cidades in REGIOES.items():
        for c in cidades:
            if _norm(c) == nome_norm:
                return reg
    return None

def next_se(se):
    y, w = divmod(int(se), 100)
    w += 1
    if w > 52: w, y = 1, y + 1
    return y * 100 + w

# ── InfoDengue ────────────────────────────────────────────────────────────────
def fetch_infodengue(geocode, disease):
    qs  = (f"?geocode={geocode}&disease={disease}&format=json"
           f"&ew_start=1&ew_end=52&ey_start=2017&ey_end=2026")
    url = f"https://info.dengue.mat.br/api/alertcity{qs}"
    hdrs = {**HEADERS, "Accept-Encoding": "gzip, deflate"}
    req  = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
        return json.loads(_decode_response(r))

def _fetch_one(args):
    gc, disease = args
    try:
        return gc, fetch_infodengue(gc, disease)
    except Exception as e:
        print(f"  \033[33m[aviso]\033[0m {gc}: {e}")
        return gc, None

# ── SARIMA cascata ────────────────────────────────────────────────────────────
def sarima_forecast(cases, H=12):
    import warnings
    y    = np.array(cases, dtype=float)
    y_sm = np.where(y == 0, 0.1, y)
    n    = len(y)

    def _ic(res, fc):
        try:
            ci = res.get_forecast(steps=H).conf_int(alpha=0.10)
            lo = [max(0,int(round(float(v)))) for v in ci.iloc[:,0]]
            hi = [int(round(float(v))) for v in ci.iloc[:,1]]
            return lo, hi
        except Exception:
            sig = float(np.sqrt(np.nanmean(np.array(res.resid)**2)))
            lo = [max(0, fc[i]-int(round(1.645*sig*(i+1)**.5))) for i in range(H)]
            hi = [fc[i]+int(round(1.645*sig*(i+1)**.5)) for i in range(H)]
            return lo, hi

    def _metrics(res):
        try:
            resid = np.array(res.resid, dtype=float)
            fv    = np.array(res.fittedvalues, dtype=float)
            ml    = min(len(y_sm), len(fv))
            ya, yf = y_sm[-ml:], fv[-ml:]
            ss_r = float(np.nansum((ya-yf)**2))
            ss_t = float(np.nansum((ya-np.nanmean(ya))**2))
            return (round(1-ss_r/ss_t,4) if ss_t>0 else None,
                    round(float(np.sqrt(np.nanmean(resid**2))),4),
                    round(float(np.nanmean(np.abs(resid))),4))
        except Exception:
            return None, None, None

    def _result(res, label):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = [max(0,int(round(float(v)))) for v in res.get_forecast(steps=H).predicted_mean]
        lo, hi = _ic(res, fc)
        r2, rmse, mae = _metrics(res)
        return dict(fc=fc, lo=lo, hi=hi, model=label,
                    aic=round(float(res.aic),1), bic=round(float(res.bic),1),
                    r2=r2, rmse=rmse, mae=mae)

    def _fit(order, seas, method="lbfgs"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(y_sm, order=order, seasonal_order=seas,
                        enforce_stationarity=False, enforce_invertibility=False, trend="n")
            return m.fit(disp=False, maxiter=300, method=method)

    if n >= 104:
        for order, seas, method, lbl in [
            ((2,1,0),(1,0,0,52),"lbfgs", "SARIMA(2,1,0)(1,0,0)[52]"),
            ((1,1,1),(1,0,0,52),"powell","SARIMA(1,1,1)(1,0,0)[52]"),
            ((1,1,0),(1,0,0,52),"lbfgs", "SARIMA(1,1,0)(1,0,0)[52]"),
        ]:
            try:
                res = _fit(order, seas, method)
                print(f"  \033[32m[modelo]\033[0m {lbl} · AIC {round(float(res.aic),1)}")
                return _result(res, lbl)
            except Exception as e:
                print(f"  \033[33m[skip]\033[0m {lbl}: {e}")

    for order, lbl in [((2,1,0),"ARIMA(2,1,0)"),((1,1,1),"ARIMA(1,1,1)")]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(y_sm, order=order).fit()
            print(f"  \033[33m[modelo]\033[0m {lbl} · AIC {round(float(res.aic),1)}")
            return _result(res, lbl)
        except Exception:
            pass

    # HW fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        use_s = n >= 2*52
        hw = ExponentialSmoothing(y_sm, trend="add",
                                  seasonal="add" if use_s else None,
                                  seasonal_periods=52 if use_s else None,
                                  initialization_method="estimated").fit(optimized=True)
    fc  = [max(0,int(round(float(v)))) for v in hw.forecast(H)]
    res = np.array(hw.resid, dtype=float)
    sig = float(np.sqrt(np.nanmean(res**2)))
    lo  = [max(0, fc[i]-int(round(1.645*sig*(i+1)**.5))) for i in range(H)]
    hi  = [fc[i]+int(round(1.645*sig*(i+1)**.5)) for i in range(H)]
    fv  = np.array(hw.fittedvalues); ml = min(len(y_sm),len(fv))
    ya2, yf2 = y_sm[-ml:], fv[-ml:]
    ss_r = float(np.nansum((ya2-yf2)**2)); ss_t = float(np.nansum((ya2-np.nanmean(ya2))**2))
    aic = round(float(hw.aic),1) if hasattr(hw,"aic") else None
    print(f"  \033[33m[modelo]\033[0m Holt-Winters ETS · AIC {aic}")
    return dict(fc=fc, lo=lo, hi=hi, model="Holt-Winters ETS", aic=aic, bic=None,
                r2=round(1-ss_r/ss_t,4) if ss_t>0 else None,
                rmse=round(float(np.sqrt(np.nanmean(res**2))),4),
                mae=round(float(np.nanmean(np.abs(res))),4))

# ── Agregação e modelo regional ───────────────────────────────────────────────
def get_regional_aggregate(region, disease):
    key = (region, disease)
    with _lock:
        if key in _regional_data:
            return _regional_data[key]

    city_names = REGIOES.get(region, [])
    geocodes   = []
    missing    = []
    for name in city_names:
        gc = geocode_of_name(name)
        if gc:
            geocodes.append(gc)
        else:
            missing.append(name)
    if missing:
        print(f"  \033[31m[aviso]\033[0m Cidades não encontradas no IBGE: {missing}")

    print(f"  \033[36m[região]\033[0m {region}: buscando {len(geocodes)}/{len(city_names)} cidades…")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_fetch_one, [(g, disease) for g in geocodes]))
    print(f"  \033[36m[região]\033[0m fetch concluído em {round(time.time()-t0,1)}s")

    se_total = {}
    city_se  = {}   # geocode → {SE: casos}
    for gc, data in results:
        if not data or not isinstance(data, list):
            continue
        city_se[gc] = {}
        for row in data:
            se  = int(row["SE"])
            cas = max(0, round(float(row.get("casos_est") or row.get("casos") or 0)))
            city_se[gc][se] = cas
            se_total[se]    = se_total.get(se, 0) + cas

    ses   = sorted(se_total.keys())
    cases = [se_total[se] for se in ses]
    agg   = {"ses": ses, "cases": cases, "city_se": city_se}
    with _lock:
        _regional_data[key] = agg
    return agg

def get_regional_model(region, disease, H=12):
    key = (region, disease, H)
    with _lock:
        if key in _regional_model:
            return _regional_model[key]

    agg = get_regional_aggregate(region, disease)
    print(f"  \033[36m[sarima]\033[0m Ajustando {region}/{disease} ({len(agg['cases'])} obs)…")
    t0     = time.time()
    result = sarima_forecast(agg["cases"], H=H)
    elapsed = round(time.time()-t0,1)
    print(f"  \033[32m[ok]\033[0m {result['model']} · AIC {result['aic']} · {elapsed}s")

    last_se = agg["ses"][-1]
    fses = []
    for _ in range(H):
        last_se = next_se(last_se)
        fses.append(last_se)

    payload = {**result, "ses": agg["ses"], "fses": fses,
               "city_se": agg["city_se"], "region_cases": agg["cases"]}
    with _lock:
        _regional_model[key] = payload
    return payload

def build_city_payload(geocode, disease, region, H=12):
    key = (geocode, disease, H)
    with _lock:
        if key in _city_fc_cache:
            return _city_fc_cache[key]

    mdl         = get_regional_model(region, disease, H)
    ses         = mdl["ses"]
    reg_cases   = mdl["region_cases"]
    city_se     = mdl["city_se"]

    # Histórico da cidade
    gc_str      = str(geocode)
    city_cases  = [city_se.get(gc_str, {}).get(se, 0) for se in ses]

    # Verificação: cidade encontrada nos dados regionais?
    if sum(city_cases) == 0:
        print(f"  \033[31m[aviso]\033[0m geocode {geocode} não encontrado no agregado regional")
        # Tenta buscar direto
        try:
            raw = fetch_infodengue(geocode, disease)
            for row in raw:
                se  = int(row["SE"])
                cas = max(0, round(float(row.get("casos_est") or row.get("casos") or 0)))
                if se in mdl.get("ses_set", set(ses)):
                    city_cases[ses.index(se) if se in ses else -1] = cas
        except Exception:
            pass

    # Proporção sobre as últimas 2 temporadas (104 semanas)
    recent_start = max(0, len(ses) - 104)
    city_recent  = sum(city_cases[recent_start:])
    reg_recent   = sum(reg_cases[recent_start:])
    n_cities     = max(1, len(city_se))
    prop = city_recent / reg_recent if reg_recent > 0 else 1 / n_cities
    print(f"  \033[36m[prop]\033[0m {geocode} → {round(prop*100,2)}% da região {region}")

    fc = [max(0, int(round(v * prop))) for v in mdl["fc"]]
    lo = [max(0, int(round(v * prop))) for v in mdl["lo"]]
    hi = [max(0, int(round(v * prop))) for v in mdl["hi"]]

    out = {
        "historical": [{"SE": s, "casos": c} for s, c in zip(ses, city_cases)],
        "fc": fc, "lo": lo, "hi": hi, "fses": mdl["fses"],
        "model":  mdl["model"],
        "region": region,
        "prop":   round(prop, 4),
        "aic":    mdl["aic"], "bic": mdl.get("bic"),
        "r2":     mdl.get("r2"), "rmse": mdl.get("rmse"), "mae": mdl.get("mae"),
    }
    with _lock:
        _city_fc_cache[key] = out
    return out

# ── Handler ───────────────────────────────────────────────────────────────────
class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        raw  = self.path
        path = raw.split("?")[0].rstrip("/") or "/"

        if path in ("/", f"/{HTML}"):
            self.path = f"/{HTML}"; return super().do_GET()

        # Mapa — estado SP (todos os municípios como TopoJSON)
        if path == "/api/mapa/sp":
            return self._proxy(
                "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35/municipios")

        # Mapa — região intermediária (GeoJSON)
        if path.startswith("/api/malha/regiao/"):
            rid = path.rsplit("/",1)[-1]
            return self._proxy(
                f"https://servicodados.ibge.gov.br/api/v3/malhas/regioes-intermediarias/{rid}?formato=application/json")

        # Mapa — municipio individual (GeoJSON)
        if path.startswith("/api/malha/municipio/"):
            mid = path.rsplit("/",1)[-1]
            return self._proxy(
                f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{mid}?formato=application/json")

        # InfoDengue proxy
        if path == "/api/infodengue":
            qs = raw[len("/api/infodengue"):]
            return self._proxy(f"https://info.dengue.mat.br/api/alertcity{qs}")

        # Forecast endpoint
        if path == "/api/forecast":
            return self._forecast(raw)

        FIXED = {
            "/api/municipios/35":   "https://servicodados.ibge.gov.br/api/v1/localidades/estados/35/municipios",
            "/api/malha/estado/35": "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/json",
        }
        if path in FIXED:
            return self._proxy(FIXED[path])
        return super().do_GET()

    def _forecast(self, raw_path):
        if not HAS_SARIMA:
            return self._json(503, {"erro":"statsmodels não instalado.","fix":"pip install statsmodels numpy"})

        qs  = urllib.parse.parse_qs(raw_path.split("?",1)[-1] if "?" in raw_path else "")
        gc  = qs.get("geocode",[""])[0]
        dis = qs.get("disease",["dengue"])[0]
        H   = int(qs.get("H",["12"])[0])
        reg = qs.get("region",[""])[0]

        if not gc:
            return self._json(400, {"erro":"geocode obrigatorio"})
        if not reg:
            reg = region_of_geocode(gc) or ""
        if not reg:
            return self._json(422, {"erro":f"Cidade {gc} não mapeada a nenhuma região SP"})

        try:
            payload = build_city_payload(gc, dis, reg, H)
            return self._json(200, payload)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\033[31m[erro]\033[0m\n{tb}")
            return self._json(500, {"erro": str(e)})

    def _proxy(self, url):
        try:
            hdrs = {**HEADERS, "Accept-Encoding": "gzip, deflate"}
            req  = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                raw = resp.read()
                enc = resp.headers.get("Content-Encoding","")
                ct  = resp.headers.get("Content-Type","application/json")
                import gzip as _gz
                if enc == "gzip" or (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
                    try: raw = _gz.decompress(raw)
                    except Exception: pass
                self._ok(ct, raw)
        except urllib.error.HTTPError as e:
            self._json(e.code, {"erro": f"upstream HTTP {e.code}"})
        except Exception as e:
            self._json(502, {"erro": str(e)})

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers(); self.wfile.write(body)

    def _ok(self, ct, body):
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers(); self.wfile.write(body)

    def log_message(self, fmt, *args):
        try:
            first = args[0] if args else ""
            if isinstance(first, str) and " " in first:
                parts = first.strip('"').split()
                path  = parts[1] if len(parts) > 1 else first
                status= str(args[1]) if len(args) > 1 else "?"
            else:
                status, path = str(first), str(args[1]) if len(args) > 1 else ""
            if path == "/favicon.ico": return
            cor = "\033[32m" if status.startswith("2") else "\033[33m" if status.startswith("3") else "\033[31m"
            print(f"  {cor}{status}\033[0m  {path}")
        except Exception:
            pass

# ── Main ──────────────────────────────────────────────────────────────────────
def testar():
    for nome, url in [
        ("IBGE",      "https://servicodados.ibge.gov.br/api/v1/localidades/estados/35/municipios"),
        ("InfoDengue","https://info.dengue.mat.br/api/alertcity?geocode=3550308&disease=dengue&format=json&ew_start=1&ew_end=1&ey_start=2023&ey_end=2023"),
    ]:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as r:
                r.read(128)
            print(f"  \033[32m✓\033[0m  {nome}")
        except Exception as e:
            print(f"  \033[31m✗\033[0m  {nome}  →  {e}")

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    if not os.path.exists(HTML):
        print(f"\n  [ERRO] '{HTML}' não encontrado em {base}\n"); sys.exit(1)
    sm = "\033[32minstalado\033[0m" if HAS_SARIMA else "\033[31mNÃO instalado — pip install statsmodels numpy\033[0m"
    print()
    print("  ┌──────────────────────────────────────────────┐")
    print("  │   Dashboard  Dengue · Chikungunya · Zika     │")
    print("  │   Modelo por REGIÃO — SARIMA regional        │")
    print("  ├──────────────────────────────────────────────┤")
    print(f"  │   http://localhost:{PORTA}                       │")
    print("  │   Ctrl+C para encerrar                       │")
    print("  └──────────────────────────────────────────────┘")
    print(f"\n  statsmodels : {sm}")
    print("\n  Testando conectividade…")
    testar()
    print("\n  \033[33mNota:\033[0m 1ª busca por região agrega ~30-90 cidades em paralelo (10-30s).\n")
    threading.Timer(0.9, lambda: webbrowser.open(f"http://localhost:{PORTA}")).start()
    with http.server.ThreadingHTTPServer(("", PORTA), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n  Encerrado.\n")

if __name__ == "__main__":
    main()
