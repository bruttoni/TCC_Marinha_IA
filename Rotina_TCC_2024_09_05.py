#################################################################################
#                        ########################################################
#Deteccao de emergencias ########################################################
#                        ########################################################
#################################################################################

#################################################################################
#Explicacoes#####################################################################
#################################################################################

#local: complexo de alimentacao da marinha (so serve almoco)
    #espaco 1111: outros espacos
    #espaco 2222: cozinha
    #espaco 3333: refeitorio
    #espaco 4444: banheiro
    #espaco 5555: vestiario/banheiro

#pessoas: cinco tipos
    #tipo 1000: comensais (300 desse tipo)
    #tipo 2000: cozinheiros - ficam na cozinha / cozinhando (15 desse tipo)
    #tipo 3000: cozinheiros - ficam no refeitório / repondo comida / servindo a fila (5 desse tipo)  
    #tipo 4000: pessoal da limpeza / fim do dia (5 desse tipo)
    #tipo 5000: pessoal da limpeza / durante o dia (2 desse tipo)

#para cada tipo de pessoa serao considerados tres comportamentos
    #comportamento tradicional ou rotineiro (sit_rh = 1)
    #comportamento atipico (doenca ou ferias) (sit_rh = 2 ou 3)
    #comportamento de emergencia (situacao de emergencia) (sit_rh não aplicável)

#periodo do modelo: dias uteis de 2020 a 2023 (1014 dias uteis)
#numero de dias de emergencia: 40 dias de emergencia (media de 10 por ano)

#################################################################################
#instalando e importando pacotes#################################################
#################################################################################

import pandas as pd
import holidays
import numpy as np
import random
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#################################################################################
#Criando a base de dados#########################################################
#################################################################################

#criando um vetor contendo todos os dias uteis de 2020 a 2023
#################################################################################

br_holidays = holidays.Brazil(years=range(2020, 2024))
date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
business_days = date_range[~date_range.isin(br_holidays)]

#criando as datas com emergencia
#################################################################################

#escolhendo 40 dias de emergencia e criando um dataframe com esses dias
emergencia = random.sample(list(business_days), 40)
df_emergencia = pd.DataFrame(emergencia, columns=['emergencia'])

#criando a hora de inicio da emergencia
start_time = 7  
end_time = 20   
start_time_in_days = start_time / 24
end_time_in_days = end_time / 24
random_times = np.random.uniform(start_time_in_days, end_time_in_days, df_emergencia.shape[0])
df_emergencia['hr_inicio_emergencia'] = pd.to_datetime(df_emergencia['emergencia']) + pd.to_timedelta(random_times, unit='D')

#criando a hora de fim da emergencia
durations_in_hours = np.random.uniform(1, 12, df_emergencia.shape[0])
durations_in_days = durations_in_hours / 24
df_emergencia['hr_fim_emergencia'] = df_emergencia['hr_inicio_emergencia'] + pd.to_timedelta(durations_in_days, unit='D')
df_emergencia['hr_fim_emergencia'] = df_emergencia.apply(
    lambda row: min(row['hr_fim_emergencia'], row['emergencia'] + pd.Timedelta(hours=23.9999)),
    axis=1
)

#Criando 300 individuos do tipo 1000 (todos os comportamentos possiveis)
#################################################################################

# Dataframe para armazenar os DataFrames de cada indivíduo (tipo = 1000)
df_t1000 = pd.DataFrame()

# Loop para gerar dados para cada indivíduo de 1001 a 1300
for i in range(1001, 1301):
    
    # Criando o comportamento de rotina para cada individuo (tipo = 1000)
    
    df_individuo = pd.DataFrame(business_days, columns=['data'])
    df_individuo['cod_individuo'] = i
    df_individuo['tipo_individuo'] = 1000

    start_time = 11.5  
    end_time = 12.5   
    start_time_in_days = start_time / 24
    end_time_in_days = end_time / 24
    random_times = np.random.uniform(start_time_in_days, end_time_in_days, df_individuo.shape[0])
    df_individuo['hr_saida_1'] = pd.to_datetime(df_individuo['data']) + pd.to_timedelta(random_times, unit='D')
    df_individuo['vlc_saida_1'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_1'] = 'Outros Espaços'
    df_individuo['cod_espaco_1'] = '1111'
    
    df_individuo['hr_entrada_2'] = df_individuo['hr_saida_1']
    df_individuo['vlc_entrada_2'] = df_individuo['vlc_saida_1']
    min_additional_time = 20 / 60
    max_additional_time = 45 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_2'] = df_individuo['hr_entrada_2'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_2'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_2'] = 'Refeitório'
    df_individuo['cod_espaco_2'] = '3333'
    
    df_individuo['hr_entrada_3'] = df_individuo['hr_saida_2']
    df_individuo['vlc_entrada_3'] = df_individuo['vlc_saida_2']
    min_additional_time = 5 / 60
    max_additional_time = 15 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_3'] = df_individuo['hr_entrada_3'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_3'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_3'] = 'Banheiro'
    df_individuo['cod_espaco_3'] = '4444'
    
    df_individuo['hr_entrada_4'] = df_individuo['hr_saida_3']
    df_individuo['vlc_entrada_4'] = df_individuo['vlc_saida_3']
    min_additional_time = 2 / 60
    max_additional_time = 10 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_4'] = df_individuo['hr_entrada_4'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_4'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_4'] = 'Refeitório'
    df_individuo['cod_espaco_4'] = '3333'
    
    df_individuo['hr_entrada_5'] = df_individuo['hr_saida_4']
    df_individuo['vlc_entrada_5'] = df_individuo['vlc_saida_4']
    df_individuo['nome_espaco_5'] = 'Outros Espaços'
    df_individuo['cod_espaco_5'] = '1111'

    # Criando comportamento de emergência para cada individuo (tipo = 1000)
    
    df_emerg_individuo = pd.merge(df_emergencia, df_individuo, left_on='emergencia', right_on='data', how='inner')

    def get_location_at_emergency(row):
        emergency_time = row['hr_inicio_emergencia']
        if emergency_time >= row['hr_entrada_5']:
            return '5'
        elif emergency_time >= row['hr_entrada_4']:
            return '4'
        elif emergency_time >= row['hr_entrada_3']:
            return '3'
        elif emergency_time >= row['hr_entrada_2']:
            return '2'
        else:
            return '1'

    df_emerg_individuo['loc_mom_emerg'] = df_emerg_individuo.apply(get_location_at_emergency, axis=1)

    min_additional_time = 0 / 60
    max_additional_time = 5 / 60
    atraso_reacao_emerg = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['atraso_reacao_emerg'] = pd.to_timedelta(atraso_reacao_emerg, unit='H')
    df_emerg_individuo['vlc_loc_emerg'] = np.random.uniform(9, 12, df_emerg_individuo.shape[0])
    
    def funcao_hr_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_saida_1']

    df_emerg_individuo['hr_saida_emerg_1'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_1, axis=1)

    def funcao_vlc_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_saida_1']

    df_emerg_individuo['vlc_saida_emerg_1'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_1, axis=1)

    def funcao_nome_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_1']

    df_emerg_individuo['nome_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_1, axis=1)

    def funcao_cod_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_1']

    df_emerg_individuo['cod_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_1, axis=1)

    def funcao_hr_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_entrada_2']

    df_emerg_individuo['hr_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_2, axis=1)

    def funcao_vlc_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_entrada_2']

    df_emerg_individuo['vlc_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_2, axis=1)

    def funcao_hr_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_2'])
        else:
            return row['hr_saida_2']

    df_emerg_individuo['hr_saida_emerg_2'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_2, axis=1)

    def funcao_vlc_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_2']

    df_emerg_individuo['vlc_saida_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_2, axis=1)

    def funcao_nome_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
           return 'N/A'
        else:
           return row['nome_espaco_2']

    df_emerg_individuo['nome_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_2, axis=1)

    def funcao_cod_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_2']

    df_emerg_individuo['cod_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_2, axis=1)

    def funcao_hr_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['hr_saida_emerg_2']
        else:
            return row['hr_entrada_3']

    df_emerg_individuo['hr_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_3, axis=1)

    def funcao_vlc_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_3']

    df_emerg_individuo['vlc_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_3, axis=1)

    def funcao_hr_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_3'])
        else:
            return row['hr_saida_3']

    df_emerg_individuo['hr_saida_emerg_3'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_3, axis=1)

    def funcao_vlc_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_3']

    df_emerg_individuo['vlc_saida_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_3, axis=1)

    def funcao_nome_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_3']

    df_emerg_individuo['nome_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_3, axis=1)

    def funcao_cod_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return '1111'
        else:
            return row['cod_espaco_3']

    df_emerg_individuo['cod_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_3, axis=1)

    def funcao_hr_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_3']
        else:
            return row['hr_entrada_4']

    df_emerg_individuo['hr_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_4, axis=1)

    def funcao_vlc_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_4']

    df_emerg_individuo['vlc_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_4, axis=1)

    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo['hr_entrada_emerg_4'] + pd.to_timedelta(additional_times, unit='H')

    def funcao_hr_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_saida_emerg_4'], row['hr_saida_4'])
        elif emergency_loc == '4':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_4'])
        else:
            return row['hr_saida_4']

    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_4, axis=1)

    def funcao_vlc_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc =='4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_4']

    df_emerg_individuo['vlc_saida_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_4, axis=1)

    def funcao_nome_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['nome_espaco_4']

    df_emerg_individuo['nome_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_4, axis=1)

    def funcao_cod_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['cod_espaco_4']

    df_emerg_individuo['cod_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_4, axis=1)

    def funcao_hr_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_4']
        elif emergency_loc == '4':
            return row['hr_saida_emerg_4']
        else:
            return row['hr_entrada_5']

    df_emerg_individuo['hr_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_5, axis=1)

    def funcao_vlc_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc == '4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_5']

    df_emerg_individuo['vlc_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_5, axis=1)

    def funcao_nome_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['nome_espaco_5']

    df_emerg_individuo['nome_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_5, axis=1)

    def funcao_cod_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['cod_espaco_5']

    df_emerg_individuo['cod_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_5, axis=1)

    for j in range(1, 5):
        df_emerg_individuo[f'hr_saida_{j}'] = df_emerg_individuo[f'hr_saida_emerg_{j}']
        df_emerg_individuo[f'vlc_saida_{j}'] = df_emerg_individuo[f'vlc_saida_emerg_{j}']

    for j in range(2, 6):
        df_emerg_individuo[f'hr_entrada_{j}'] = df_emerg_individuo[f'hr_entrada_emerg_{j}']
        df_emerg_individuo[f'vlc_entrada_{j}'] = df_emerg_individuo[f'vlc_entrada_emerg_{j}']

    for j in range(1, 6):
        df_emerg_individuo[f'nome_espaco_{j}'] = df_emerg_individuo[f'nome_espaco_emerg_{j}']
        df_emerg_individuo[f'cod_espaco_{j}'] = df_emerg_individuo[f'cod_espaco_emerg_{j}']

    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[0:3], axis=1)
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[29:], axis=1)

    #Criando comportamento em dia de doenca para cada individuo (tipo = 1000)
    
    sick_days_individuo = random.sample(list(business_days), 2)

    df_sick_days_individuo = pd.DataFrame(sick_days_individuo, columns=['data'])

    df_sick_days_individuo['cod_individuo'] = i
    df_sick_days_individuo['tipo_individuo'] = 1000

    for j in range(1, 5):
        df_sick_days_individuo[f'hr_saida_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_saida_{j}'] = np.nan

    for j in range(2, 6):
        df_sick_days_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_entrada_{j}'] = np.nan

    for j in range(1, 6):
        df_sick_days_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_sick_days_individuo[f'cod_espaco_{j}'] = 'N/A'

    #Criando comportamento de ferias para cada individuo (tipo = 1000)

    def sortear_ferias(dias_uteis, ano):
        dias_uteis_ano = dias_uteis[dias_uteis.year == ano]

        midyear_candidates = dias_uteis_ano[
            ((dias_uteis_ano.month == 7) | (dias_uteis_ano.month == 8)) & #julho ou agosto
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]

        endyear_candidates = dias_uteis_ano[
            (dias_uteis_ano.month == 12) | (dias_uteis_ano.month == 1) & #dezembro ou janeiro
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
    
        midyear = random.choice(midyear_candidates)
        endyear = random.choice(endyear_candidates)

        return midyear, endyear

    ferias = []
    for ano in range(2020, 2024):
        ferias.append(sortear_ferias(business_days, ano))

    df_ferias = pd.DataFrame(ferias, columns=['inicio_ferias_meioano', 'inicio_ferias_finalano'])

    df_ferias_empilhado = pd.melt(df_ferias, value_name='inicio_ferias')[['inicio_ferias']]

    df_ferias_empilhado = df_ferias_empilhado.sort_values(by='inicio_ferias').reset_index(drop=True)

    df_ferias_empilhado['termino_ferias'] = df_ferias_empilhado['inicio_ferias'] + pd.Timedelta(days=14)

    def get_business_days_between(start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        dias_ut = date_range[date_range.isin(business_days)]
        return dias_ut

    df_ferias_empilhado['data'] = df_ferias_empilhado.apply(
        lambda row: get_business_days_between(row['inicio_ferias'], row['termino_ferias']),
        axis=1
        )

    df_periodo_ferias_1 = df_ferias_empilhado.explode('data').reset_index(drop=True)
    
    df_ferias_individuo = df_periodo_ferias_1[['data']].dropna().reset_index(drop=True)

    df_ferias_individuo.columns = ['data']
    
    df_ferias_individuo = df_ferias_individuo.drop_duplicates()

    df_ferias_individuo['cod_individuo'] = i
    df_ferias_individuo['tipo_individuo'] = 1000

    for j in range(1, 5):
        df_ferias_individuo[f'hr_saida_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_saida_{j}'] = np.nan

    for j in range(2, 6):
        df_ferias_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_entrada_{j}'] = np.nan

    for j in range(1, 6):
        df_ferias_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_ferias_individuo[f'cod_espaco_{j}'] = 'N/A'
        
    #Juntando todos os comportamentos para cada individuo (tipo = 1000)

    emergencia_datas = df_emerg_individuo['data']
    df_individuo_semerg = df_individuo[~df_individuo['data'].isin(emergencia_datas)]
    df_individuo_emerg = pd.concat([df_individuo_semerg, df_emerg_individuo], ignore_index=True)

    sick_days_datas = df_sick_days_individuo['data']
    df_individuo_emerg_sdoenca = df_individuo_emerg[~df_individuo_emerg['data'].isin(sick_days_datas)]
    df_individuo_emerg_doenca = pd.concat([df_individuo_emerg_sdoenca, df_sick_days_individuo], ignore_index=True)

    ferias_datas = df_ferias_individuo['data']
    df_individuo_emerg_doenca_sferias = df_individuo_emerg_doenca[~df_individuo_emerg_doenca['data'].isin(ferias_datas)]
    df_ind_final = pd.concat([df_individuo_emerg_doenca_sferias, df_ferias_individuo], ignore_index=True)

    # Criando variavel 'ind_emergencia'
    df_ind_final['ind_emergencia'] = df_ind_final['data'].isin(df_emergencia['emergencia']).astype(int)

    # Criando variavel 'sit_rh'
    df_ind_final['sit_rh'] = 1 # 1 = (em serviço)
    df_ind_final.loc[df_ind_final['data'].isin(df_sick_days_individuo['data']), 'sit_rh'] = 2 # 2 = (doente -> altera 'em serviço')
    df_ind_final.loc[(df_ind_final['data'].isin(df_ferias_individuo['data'])), 'sit_rh'] = 3 # 3 = (ferias -> altera 'em serviço' e 'doente')

    # Ordenando o DataFrame
    df_ind_final = df_ind_final.sort_values(by='data').reset_index(drop=True)

    # Salvando o DataFrame no dicionário
    df_t1000 = pd.concat([df_t1000, df_ind_final], ignore_index=True)

#limpando a memoria
#################################################################################################

# Listando todos os objetos na memória
objetos_a_manter = ['business_days', 'df_emergencia', 'df_t1000', 'objetos_a_manter', 'pd', 'holidays', 'np', 'random', 'sm', 'LogisticRegression', 'cross_val_score', 'KFold', 'accuracy_score']

# Removendo todos os objetos exceto os que estão na lista de objetos a manter
for obj in dir():
    if obj not in objetos_a_manter and not obj.startswith("__"):
        del globals()[obj]
        
# Removendo as variáveis 'objetos_a_manter' e 'obj'
del objetos_a_manter
del obj

#Criando 15 individuos do tipo 2000 (todos os comportamentos possiveis)
#################################################################################

# Dataframe para armazenar os DataFrames de cada indivíduo (tipo = 2000)
df_t2000 = pd.DataFrame()

# Loop para gerar dados para cada indivíduo de 2001 a 2015
for i in range(2001, 2016):

    # Criando o comportamento de rotina para cada individuo (tipo = 2000)

    df_individuo = pd.DataFrame(business_days, columns=['data'])
    df_individuo['cod_individuo'] = i
    df_individuo['tipo_individuo'] = 2000

    start_time = 6.5  
    end_time = 7.5   
    start_time_in_days = start_time / 24
    end_time_in_days = end_time / 24
    random_times = np.random.uniform(start_time_in_days, end_time_in_days, df_individuo.shape[0])
    df_individuo['hr_saida_1'] = pd.to_datetime(df_individuo['data']) + pd.to_timedelta(random_times, unit='D')
    df_individuo['vlc_saida_1'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_1'] = 'Outros Espaços'
    df_individuo['cod_espaco_1'] = '1111'
    
    df_individuo['hr_entrada_2'] = df_individuo['hr_saida_1']
    df_individuo['vlc_entrada_2'] = df_individuo['vlc_saida_1']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_2'] = df_individuo['hr_entrada_2'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_2'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_2'] = 'Cozinha'
    df_individuo['cod_espaco_2'] = '2222'
    
    df_individuo['hr_entrada_3'] = df_individuo['hr_saida_2']
    df_individuo['vlc_entrada_3'] = df_individuo['vlc_saida_2']
    min_additional_time = 15 / 60
    max_additional_time = 30 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_3'] = df_individuo['hr_entrada_3'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_3'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_3'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_3'] = '5555'
    
    df_individuo['hr_entrada_4'] = df_individuo['hr_saida_3']
    df_individuo['vlc_entrada_4'] = df_individuo['vlc_saida_3']
    min_additional_time = 150 / 60
    max_additional_time = 180 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_4'] = df_individuo['hr_entrada_4'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_4'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_4'] = 'Cozinha'
    df_individuo['cod_espaco_4'] = '2222'
    
    df_individuo['hr_entrada_5'] = df_individuo['hr_saida_4']
    df_individuo['vlc_entrada_5'] = df_individuo['vlc_saida_4']
    min_additional_time = 15 / 60
    max_additional_time = 30 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_5'] = df_individuo['hr_entrada_5'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_5'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_5'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_5'] = '5555'
    
    df_individuo['hr_entrada_6'] = df_individuo['hr_saida_5']
    df_individuo['vlc_entrada_6'] = df_individuo['vlc_saida_5']
    additional_time = 465 / 60
    df_individuo['hr_saida_6'] = df_individuo['hr_entrada_2'] + pd.to_timedelta(additional_time, unit='H')
    df_individuo['vlc_saida_6'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_6'] = 'Cozinha'
    df_individuo['cod_espaco_6'] = '2222'
    
    df_individuo['hr_entrada_7'] = df_individuo['hr_saida_6']
    df_individuo['vlc_entrada_7'] = df_individuo['vlc_saida_6']
    min_additional_time = 15 / 60
    max_additional_time = 30 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_7'] = df_individuo['hr_entrada_7'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_7'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_7'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_7'] = '5555'
    
    df_individuo['hr_entrada_8'] = df_individuo['hr_saida_7']
    df_individuo['vlc_entrada_8'] = df_individuo['vlc_saida_7']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_8'] = df_individuo['hr_entrada_8'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_8'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_8'] = 'Cozinha'
    df_individuo['cod_espaco_8'] = '2222'
    
    df_individuo['hr_entrada_9'] = df_individuo['hr_saida_8']
    df_individuo['vlc_entrada_9'] = df_individuo['vlc_saida_8']
    df_individuo['nome_espaco_9'] = 'Outros Espaços'
    df_individuo['cod_espaco_9'] = '1111'
    
    # Criando comportamento de emergência para cada individuo (tipo = 2000)
    
    df_emerg_individuo = pd.merge(df_emergencia, df_individuo, left_on='emergencia', right_on='data', how='inner')
    
    def get_location_at_emergency(row):
        emergency_time = row['hr_inicio_emergencia']
        if emergency_time >= row['hr_entrada_9']:
            return '9'
        elif emergency_time >= row['hr_entrada_8']:
            return '8'
        elif emergency_time >= row['hr_entrada_7']:
            return '7'
        elif emergency_time >= row['hr_entrada_6']:
            return '6'
        elif emergency_time >= row['hr_entrada_5']:
            return '5'
        elif emergency_time >= row['hr_entrada_4']:
            return '4'
        elif emergency_time >= row['hr_entrada_3']:
            return '3'
        elif emergency_time >= row['hr_entrada_2']:
            return '2'
        else:
            return '1'

    df_emerg_individuo['loc_mom_emerg'] = df_emerg_individuo.apply(get_location_at_emergency, axis=1)
    
    min_additional_time = 0 / 60
    max_additional_time = 5 / 60
    atraso_reacao_emerg = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['atraso_reacao_emerg'] = pd.to_timedelta(atraso_reacao_emerg, unit='H')
    df_emerg_individuo['vlc_loc_emerg'] = np.random.uniform(9, 12, df_emerg_individuo.shape[0])
        
    def funcao_hr_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_saida_1']
        
    df_emerg_individuo['hr_saida_emerg_1'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_1, axis=1)
        
    def funcao_vlc_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_saida_1']
            
    df_emerg_individuo['vlc_saida_emerg_1'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_1, axis=1)

    def funcao_nome_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_1']

    df_emerg_individuo['nome_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_1, axis=1)

    def funcao_cod_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_1']

    df_emerg_individuo['cod_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_1, axis=1)
    
    def funcao_hr_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_entrada_2']
        
    df_emerg_individuo['hr_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_2, axis=1)
        
    def funcao_vlc_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_entrada_2']
            
    df_emerg_individuo['vlc_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_2, axis=1)
            
    def funcao_hr_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_2'])
        else:
            return row['hr_saida_2']

    df_emerg_individuo['hr_saida_emerg_2'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_2, axis=1)

    def funcao_vlc_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_2']

    df_emerg_individuo['vlc_saida_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_2, axis=1)

    def funcao_nome_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_2']

    df_emerg_individuo['nome_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_2, axis=1)

    def funcao_cod_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_2']

    df_emerg_individuo['cod_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_2, axis=1)

    def funcao_hr_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['hr_saida_emerg_2']
        else:
            return row['hr_entrada_3']

    df_emerg_individuo['hr_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_3, axis=1)

    def funcao_vlc_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_3']

    df_emerg_individuo['vlc_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_3, axis=1)

    def funcao_hr_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_3'])
        else:
            return row['hr_saida_3']

    df_emerg_individuo['hr_saida_emerg_3'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_3, axis=1)

    def funcao_vlc_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_3']

    df_emerg_individuo['vlc_saida_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_3, axis=1)

    def funcao_nome_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_3']

    df_emerg_individuo['nome_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_3, axis=1)

    def funcao_cod_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return '1111'
        else:
            return row['cod_espaco_3']

    df_emerg_individuo['cod_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_3, axis=1)

    def funcao_hr_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_3']
        else:
            return row['hr_entrada_4']

    df_emerg_individuo['hr_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_4, axis=1)

    def funcao_vlc_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_4']

    df_emerg_individuo['vlc_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_4, axis=1)

    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo['hr_entrada_emerg_4'] + pd.to_timedelta(additional_times, unit='H')

    def funcao_hr_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_saida_emerg_4'], row['hr_saida_4'])
        elif emergency_loc == '4':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_4'])
        else:
            return row['hr_saida_4']

    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_4, axis=1)

    def funcao_vlc_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc =='4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_4']

    df_emerg_individuo['vlc_saida_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_4, axis=1)

    def funcao_nome_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['nome_espaco_4']

    df_emerg_individuo['nome_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_4, axis=1)

    def funcao_cod_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['cod_espaco_4']

    df_emerg_individuo['cod_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_4, axis=1)

    def funcao_hr_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_4']
        elif emergency_loc == '4':
            return row['hr_saida_emerg_4']
        else:
            return row['hr_entrada_5']

    df_emerg_individuo['hr_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_5, axis=1)

    def funcao_vlc_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc == '4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_5']

    df_emerg_individuo['vlc_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_5, axis=1)

    def funcao_hr_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_5'])
        else:
            return row['hr_saida_5']

    df_emerg_individuo['hr_saida_emerg_5'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_5, axis=1)

    def funcao_vlc_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_5']

    df_emerg_individuo['vlc_saida_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_5, axis=1)

    def funcao_nome_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'Outros Espaços'
        elif emergency_loc == '4':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_5']

    df_emerg_individuo['nome_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_5, axis=1)

    def funcao_cod_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return '1111'
        elif emergency_loc == '4':
            return '1111'
        else:
            return row['cod_espaco_5']

    df_emerg_individuo['cod_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_5, axis=1)

    def funcao_hr_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['hr_saida_emerg_5']
        else:
            return row['hr_entrada_6']

    df_emerg_individuo['hr_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_6, axis=1)

    def funcao_vlc_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_6']

    df_emerg_individuo['vlc_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_6, axis=1)

    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_6'] = df_emerg_individuo['hr_entrada_emerg_6'] + pd.to_timedelta(additional_times, unit='H')

    def funcao_hr_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.minimum(row['hr_saida_emerg_6'], row['hr_saida_6'])
        elif emergency_loc == '6':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_6'])
        else:
            return row['hr_saida_6']

    df_emerg_individuo['hr_saida_emerg_6'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_6, axis=1)

    def funcao_vlc_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_6']

    df_emerg_individuo['vlc_saida_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_6, axis=1)

    def funcao_nome_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'Cozinha'
        elif emergency_loc == '6':
            return 'Cozinha'
        else:
            return row['nome_espaco_6']

    df_emerg_individuo['nome_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_6, axis=1)

    def funcao_cod_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return '2222'
        elif emergency_loc == '6':
            return '2222'
        else:
            return row['cod_espaco_6']

    df_emerg_individuo['cod_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_6, axis=1)

    def funcao_hr_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['hr_saida_emerg_6']
        elif emergency_loc == '6':
            return row['hr_saida_emerg_6']
        else:
            return row['hr_entrada_7']

    df_emerg_individuo['hr_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_7, axis=1)

    def funcao_vlc_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_7']

    df_emerg_individuo['vlc_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_7, axis=1)

    def funcao_hr_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_7'])
        else:
            return row['hr_saida_7']

    df_emerg_individuo['hr_saida_emerg_7'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_7, axis=1)

    def funcao_vlc_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_7']

    df_emerg_individuo['vlc_saida_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_7, axis=1)

    def funcao_nome_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'Outros Espaços'
        elif emergency_loc == '6':
            return 'Outros Espaços'
        elif emergency_loc == '7':
            return 'Vestiário/Banheiro'
        else:
            return row['nome_espaco_7']

    df_emerg_individuo['nome_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_7, axis=1)

    def funcao_cod_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return '1111'
        elif emergency_loc == '6':
            return '1111'
        elif emergency_loc == '7':
            return '5555'
        else:
            return row['cod_espaco_7']

    df_emerg_individuo['cod_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_7, axis=1)

    def funcao_hr_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['hr_saida_emerg_7']
        else:
            return row['hr_entrada_8']

    df_emerg_individuo['hr_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_8, axis=1)

    def funcao_vlc_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_8']

    df_emerg_individuo['vlc_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_8, axis=1)

    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_8'] = df_emerg_individuo['hr_entrada_emerg_8'] + pd.to_timedelta(additional_times, unit='H')

    def funcao_hr_saida_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.minimum(row['hr_saida_emerg_8'], row['hr_saida_8'])
        elif emergency_loc == '8':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_8'])
        else:
            return row['hr_saida_8']

    df_emerg_individuo['hr_saida_emerg_8'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_8, axis=1)

    def funcao_vlc_saida_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        elif emergency_loc == '8':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_8']

    df_emerg_individuo['vlc_saida_emerg_8'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_8, axis=1)

    def funcao_nome_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'Cozinha'
        elif emergency_loc == '8':
            return 'Cozinha'
        else:
            return row['nome_espaco_8']

    df_emerg_individuo['nome_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_8, axis=1)

    def funcao_cod_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return '2222'
        elif emergency_loc == '8':
            return '2222'
        else:
            return row['cod_espaco_8']

    df_emerg_individuo['cod_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_8, axis=1)

    def funcao_hr_entrada_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['hr_saida_emerg_8']
        elif emergency_loc == '8':
            return row['hr_saida_emerg_8']
        else:
            return row['hr_entrada_9']

    df_emerg_individuo['hr_entrada_emerg_9'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_9, axis=1)

    def funcao_vlc_entrada_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        elif emergency_loc == '8':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_9']

    df_emerg_individuo['vlc_entrada_emerg_9'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_9, axis=1)

    def funcao_nome_espaco_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'Outros Espaços'
        elif emergency_loc == '8':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_9']

    df_emerg_individuo['nome_espaco_emerg_9'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_9, axis=1)

    def funcao_cod_espaco_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return '1111'
        elif emergency_loc == '8':
            return '1111'
        else:
            return row['cod_espaco_9']

    df_emerg_individuo['cod_espaco_emerg_9'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_9, axis=1)

    for j in range(1, 9):
        df_emerg_individuo[f'hr_saida_{j}'] = df_emerg_individuo[f'hr_saida_emerg_{j}']
        df_emerg_individuo[f'vlc_saida_{j}'] = df_emerg_individuo[f'vlc_saida_emerg_{j}']

    for j in range(2, 10):
        df_emerg_individuo[f'hr_entrada_{j}'] = df_emerg_individuo[f'hr_entrada_emerg_{j}']
        df_emerg_individuo[f'vlc_entrada_{j}'] = df_emerg_individuo[f'vlc_entrada_emerg_{j}']

    for j in range(1, 10):
        df_emerg_individuo[f'nome_espaco_{j}'] = df_emerg_individuo[f'nome_espaco_emerg_{j}']
        df_emerg_individuo[f'cod_espaco_{j}'] = df_emerg_individuo[f'cod_espaco_emerg_{j}']

    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[0:3], axis=1)
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[53:], axis=1)

    #Criando comportamento em dia de doenca para cada individuo (tipo = 2000)
 
    sick_days_individuo = random.sample(list(business_days), 2)

    df_sick_days_individuo = pd.DataFrame(sick_days_individuo, columns=['data'])

    df_sick_days_individuo['cod_individuo'] = i
    df_sick_days_individuo['tipo_individuo'] = 2000

    for j in range(1, 9):
        df_sick_days_individuo[f'hr_saida_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_saida_{j}'] = np.nan

    for j in range(2, 10):
        df_sick_days_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_entrada_{j}'] = np.nan

    for j in range(1, 10):
        df_sick_days_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_sick_days_individuo[f'cod_espaco_{j}'] = 'N/A'

    #Criando comportamento de ferias para cada individuo (tipo = 2000)

    def sortear_ferias(dias_uteis, ano):
        dias_uteis_ano = dias_uteis[dias_uteis.year == ano]

        midyear_candidates = dias_uteis_ano[
            ((dias_uteis_ano.month == 7) | (dias_uteis_ano.month == 8)) & #julho ou agosto
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]

        endyear_candidates = dias_uteis_ano[
            (dias_uteis_ano.month == 12) | (dias_uteis_ano.month == 1) & #dezembro ou janeiro
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
 
        midyear = random.choice(midyear_candidates)
        endyear = random.choice(endyear_candidates)

        return midyear, endyear

    ferias = []
    for ano in range(2020, 2024):
        ferias.append(sortear_ferias(business_days, ano))

    df_ferias = pd.DataFrame(ferias, columns=['inicio_ferias_meioano', 'inicio_ferias_finalano'])

    df_ferias_empilhado = pd.melt(df_ferias, value_name='inicio_ferias')[['inicio_ferias']]

    df_ferias_empilhado = df_ferias_empilhado.sort_values(by='inicio_ferias').reset_index(drop=True)

    df_ferias_empilhado['termino_ferias'] = df_ferias_empilhado['inicio_ferias'] + pd.Timedelta(days=14)

    def get_business_days_between(start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        dias_ut = date_range[date_range.isin(business_days)]
        return dias_ut

    df_ferias_empilhado['data'] = df_ferias_empilhado.apply(
        lambda row: get_business_days_between(row['inicio_ferias'], row['termino_ferias']),
        axis=1
        )

    df_periodo_ferias_1 = df_ferias_empilhado.explode('data').reset_index(drop=True)
 
    df_ferias_individuo = df_periodo_ferias_1[['data']].dropna().reset_index(drop=True)

    df_ferias_individuo.columns = ['data']
 
    df_ferias_individuo = df_ferias_individuo.drop_duplicates()

    df_ferias_individuo['cod_individuo'] = i
    df_ferias_individuo['tipo_individuo'] = 2000

    for j in range(1, 9):
        df_ferias_individuo[f'hr_saida_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_saida_{j}'] = np.nan

    for j in range(2, 10):
        df_ferias_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_entrada_{j}'] = np.nan

    for j in range(1, 10):
        df_ferias_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_ferias_individuo[f'cod_espaco_{j}'] = 'N/A'
     
    #Juntando todos os comportamentos para cada individuo (tipo = 2000)

    emergencia_datas = df_emerg_individuo['data']
    df_individuo_semerg = df_individuo[~df_individuo['data'].isin(emergencia_datas)]
    df_individuo_emerg = pd.concat([df_individuo_semerg, df_emerg_individuo], ignore_index=True)

    sick_days_datas = df_sick_days_individuo['data']
    df_individuo_emerg_sdoenca = df_individuo_emerg[~df_individuo_emerg['data'].isin(sick_days_datas)]
    df_individuo_emerg_doenca = pd.concat([df_individuo_emerg_sdoenca, df_sick_days_individuo], ignore_index=True)

    ferias_datas = df_ferias_individuo['data']
    df_individuo_emerg_doenca_sferias = df_individuo_emerg_doenca[~df_individuo_emerg_doenca['data'].isin(ferias_datas)]
    df_ind_final = pd.concat([df_individuo_emerg_doenca_sferias, df_ferias_individuo], ignore_index=True)

    # Criando variavel 'ind_emergencia'
    df_ind_final['ind_emergencia'] = df_ind_final['data'].isin(df_emergencia['emergencia']).astype(int)

    # Criando variavel 'sit_rh'
    df_ind_final['sit_rh'] = 1 # 1 = (em serviço)
    df_ind_final.loc[df_ind_final['data'].isin(df_sick_days_individuo['data']), 'sit_rh'] = 2 # 2 = (doente -> altera 'em serviço')
    df_ind_final.loc[(df_ind_final['data'].isin(df_ferias_individuo['data'])), 'sit_rh'] = 3 # 3 = (ferias -> altera 'em serviço' e 'doente')

    # Ordenando o DataFrame
    df_ind_final = df_ind_final.sort_values(by='data').reset_index(drop=True)

    # Salvando o DataFrame no dicionário
    df_t2000 = pd.concat([df_t2000, df_ind_final], ignore_index=True)

#limpando a memoria
#################################################################################################

# Listando todos os objetos na memória
objetos_a_manter = ['business_days', 'df_emergencia', 'df_t1000', 'df_t2000', 'objetos_a_manter', 'pd', 'holidays', 'np', 'random', 'sm', 'LogisticRegression', 'cross_val_score', 'KFold', 'accuracy_score']

# Removendo todos os objetos exceto os que estão na lista de objetos a manter
for obj in dir():
    if obj not in objetos_a_manter and not obj.startswith("__"):
        del globals()[obj]
        
# Removendo as variáveis 'objetos_a_manter' e 'obj'
del objetos_a_manter
del obj

#Criando 5 individuos do tipo 3000 (todos os comportamentos possiveis)
#################################################################################

# Dataframe para armazenar os DataFrames de cada indivíduo (tipo = 3000)
df_t3000 = pd.DataFrame()

# Loop para gerar dados para cada indivíduo de 2001 a 2015
for i in range(3001, 3006):

    # Criando o comportamento de rotina para cada individuo (tipo = 3000)
    
    df_individuo = pd.DataFrame(business_days, columns=['data'])
    df_individuo['cod_individuo'] = i
    df_individuo['tipo_individuo'] = 3000
    
    start_time = 6.5  
    end_time = 7.5   
    start_time_in_days = start_time / 24
    end_time_in_days = end_time / 24
    random_times = np.random.uniform(start_time_in_days, end_time_in_days, df_individuo.shape[0])
    df_individuo['hr_saida_1'] = pd.to_datetime(df_individuo['data']) + pd.to_timedelta(random_times, unit='D')
    df_individuo['vlc_saida_1'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_1'] = 'Outros Espaços'
    df_individuo['cod_espaco_1'] = '1111'
        
    df_individuo['hr_entrada_2'] = df_individuo['hr_saida_1']
    df_individuo['vlc_entrada_2'] = df_individuo['vlc_saida_1']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_2'] = df_individuo['hr_entrada_2'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_2'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_2'] = 'Cozinha'
    df_individuo['cod_espaco_2'] = '2222'
        
    df_individuo['hr_entrada_3'] = df_individuo['hr_saida_2']
    df_individuo['vlc_entrada_3'] = df_individuo['vlc_saida_2']
    min_additional_time = 15 / 60
    max_additional_time = 30 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_3'] = df_individuo['hr_entrada_3'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_3'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_3'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_3'] = '5555'
        
    df_individuo['hr_entrada_4'] = df_individuo['hr_saida_3']
    df_individuo['vlc_entrada_4'] = df_individuo['vlc_saida_3']
    min_additional_time = 150 / 60
    max_additional_time = 180 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_4'] = df_individuo['hr_entrada_4'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_4'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_4'] = 'Cozinha'
    df_individuo['cod_espaco_4'] = '2222'
    
    df_individuo['hr_entrada_5'] = df_individuo['hr_saida_4']
    df_individuo['vlc_entrada_5'] = df_individuo['vlc_saida_4']
    min_additional_time = 15 / 60
    max_additional_time = 30 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_5'] = df_individuo['hr_entrada_5'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_5'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_5'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_5'] = '5555'
        
    df_individuo['hr_entrada_6'] = df_individuo['hr_saida_5']
    df_individuo['vlc_entrada_6'] = df_individuo['vlc_saida_5']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_6'] = df_individuo['hr_entrada_6'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_6'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_6'] = 'Cozinha'
    df_individuo['cod_espaco_6'] = '2222'
    
    df_individuo['hr_entrada_7'] = df_individuo['hr_saida_6']
    df_individuo['vlc_entrada_7'] = df_individuo['vlc_saida_6']
    additional_times = 465 / 60
    df_individuo['hr_saida_7'] = df_individuo['hr_entrada_2'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_7'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_7'] = 'Refeitório'
    df_individuo['cod_espaco_7'] = '3333'
    
    df_individuo['hr_entrada_8'] = df_individuo['hr_saida_7']
    df_individuo['vlc_entrada_8'] = df_individuo['vlc_saida_7']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_8'] = df_individuo['hr_entrada_8'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_8'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_8'] = 'Cozinha'
    df_individuo['cod_espaco_8'] = '2222'
    
    df_individuo['hr_entrada_9'] = df_individuo['hr_saida_8']
    df_individuo['vlc_entrada_9'] = df_individuo['vlc_saida_8']
    min_additional_time = 15 / 60
    max_additional_time = 30 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_9'] = df_individuo['hr_entrada_9'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_9'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_9'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_9'] = '5555'
    
    df_individuo['hr_entrada_10'] = df_individuo['hr_saida_9']
    df_individuo['vlc_entrada_10'] = df_individuo['vlc_saida_9']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_10'] = df_individuo['hr_entrada_10'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_10'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_10'] = 'Cozinha'
    df_individuo['cod_espaco_10'] = '2222'
    
    df_individuo['hr_entrada_11'] = df_individuo['hr_saida_10']
    df_individuo['vlc_entrada_11'] = df_individuo['vlc_saida_10']
    df_individuo['nome_espaco_11'] = 'Outros Espaços'
    df_individuo['cod_espaco_11'] = '1111'
    
    # Criando comportamento de emergência para cada individuo (tipo = 3000)
        
    df_emerg_individuo = pd.merge(df_emergencia, df_individuo, left_on='emergencia', right_on='data', how='inner')
        
    def get_location_at_emergency(row):
        emergency_time = row['hr_inicio_emergencia']
        if emergency_time >= row['hr_entrada_11']:
            return '11'
        elif emergency_time >= row['hr_entrada_10']:
            return '10'
        elif emergency_time >= row['hr_entrada_9']:
            return '9'
        elif emergency_time >= row['hr_entrada_8']:
            return '8'
        elif emergency_time >= row['hr_entrada_7']:
            return '7'
        elif emergency_time >= row['hr_entrada_6']:
            return '6'
        elif emergency_time >= row['hr_entrada_5']:
            return '5'
        elif emergency_time >= row['hr_entrada_4']:
            return '4'
        elif emergency_time >= row['hr_entrada_3']:
            return '3'
        elif emergency_time >= row['hr_entrada_2']:
            return '2'
        else:
            return '1'
    
    df_emerg_individuo['loc_mom_emerg'] = df_emerg_individuo.apply(get_location_at_emergency, axis=1)
        
    min_additional_time = 0 / 60
    max_additional_time = 5 / 60
    atraso_reacao_emerg = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['atraso_reacao_emerg'] = pd.to_timedelta(atraso_reacao_emerg, unit='H')
    df_emerg_individuo['vlc_loc_emerg'] = np.random.uniform(9, 12, df_emerg_individuo.shape[0])
            
    def funcao_hr_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_saida_1']
            
    df_emerg_individuo['hr_saida_emerg_1'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_1, axis=1)
            
    def funcao_vlc_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_saida_1']
                
    df_emerg_individuo['vlc_saida_emerg_1'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_1, axis=1)
    
    def funcao_nome_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_1']
    
    df_emerg_individuo['nome_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_1, axis=1)
    
    def funcao_cod_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_1']
    
    df_emerg_individuo['cod_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_1, axis=1)
        
    def funcao_hr_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_entrada_2']
            
    df_emerg_individuo['hr_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_2, axis=1)
            
    def funcao_vlc_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_entrada_2']
                
    df_emerg_individuo['vlc_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_2, axis=1)
                
    def funcao_hr_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_2'])
        else:
            return row['hr_saida_2']
    
    df_emerg_individuo['hr_saida_emerg_2'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_2, axis=1)
    
    def funcao_vlc_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_2']
    
    df_emerg_individuo['vlc_saida_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_2, axis=1)
    
    def funcao_nome_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_2']
    
    df_emerg_individuo['nome_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_2, axis=1)
    
    def funcao_cod_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_2']
    
    df_emerg_individuo['cod_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_2, axis=1)
    
    def funcao_hr_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['hr_saida_emerg_2']
        else:
            return row['hr_entrada_3']
    
    df_emerg_individuo['hr_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_3, axis=1)
    
    def funcao_vlc_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_3']
    
    df_emerg_individuo['vlc_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_3, axis=1)
    
    def funcao_hr_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_3'])
        else:
            return row['hr_saida_3']
    
    df_emerg_individuo['hr_saida_emerg_3'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_3, axis=1)
    
    def funcao_vlc_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_3']
    
    df_emerg_individuo['vlc_saida_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_3, axis=1)
    
    def funcao_nome_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_3']
    
    df_emerg_individuo['nome_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_3, axis=1)
    
    def funcao_cod_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return '1111'
        else:
            return row['cod_espaco_3']
    
    df_emerg_individuo['cod_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_3, axis=1)
    
    def funcao_hr_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_3']
        else:
            return row['hr_entrada_4']
    
    df_emerg_individuo['hr_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_4, axis=1)
    
    def funcao_vlc_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_4']
    
    df_emerg_individuo['vlc_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_4, axis=1)
    
    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo['hr_entrada_emerg_4'] + pd.to_timedelta(additional_times, unit='H')
    
    def funcao_hr_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_saida_emerg_4'], row['hr_saida_4'])
        elif emergency_loc == '4':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_4'])
        else:
            return row['hr_saida_4']
    
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_4, axis=1)
    
    def funcao_vlc_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc =='4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_4']
    
    df_emerg_individuo['vlc_saida_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_4, axis=1)
    
    def funcao_nome_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['nome_espaco_4']
    
    df_emerg_individuo['nome_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_4, axis=1)
    
    def funcao_cod_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['cod_espaco_4']
    
    df_emerg_individuo['cod_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_4, axis=1)
    
    def funcao_hr_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_4']
        elif emergency_loc == '4':
            return row['hr_saida_emerg_4']
        else:
            return row['hr_entrada_5']
    
    df_emerg_individuo['hr_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_5, axis=1)
    
    def funcao_vlc_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc == '4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_5']
    
    df_emerg_individuo['vlc_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_5, axis=1)
    
    def funcao_hr_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_5'])
        else:
            return row['hr_saida_5']
    
    df_emerg_individuo['hr_saida_emerg_5'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_5, axis=1)
    
    def funcao_vlc_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_5']
    
    df_emerg_individuo['vlc_saida_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_5, axis=1)
    
    def funcao_nome_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'Outros Espaços'
        elif emergency_loc == '4':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_5']
    
    df_emerg_individuo['nome_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_5, axis=1)
    
    def funcao_cod_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return '1111'
        elif emergency_loc == '4':
            return '1111'
        else:
            return row['cod_espaco_5']
    
    df_emerg_individuo['cod_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_5, axis=1)
    
    def funcao_hr_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['hr_saida_emerg_5']
        else:
            return row['hr_entrada_6']
    
    df_emerg_individuo['hr_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_6, axis=1)
    
    def funcao_vlc_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_6']
    
    df_emerg_individuo['vlc_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_6, axis=1)
    
    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_6'] = df_emerg_individuo['hr_entrada_emerg_6'] + pd.to_timedelta(additional_times, unit='H')
    
    def funcao_hr_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.minimum(row['hr_saida_emerg_6'], row['hr_saida_6'])
        elif emergency_loc == '6':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_6'])
        else:
            return row['hr_saida_6']
    
    df_emerg_individuo['hr_saida_emerg_6'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_6, axis=1)
    
    def funcao_vlc_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_6']
    
    df_emerg_individuo['vlc_saida_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_6, axis=1)
    
    def funcao_nome_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'Cozinha'
        elif emergency_loc == '6':
            return 'Cozinha'
        else:
            return row['nome_espaco_6']
    
    df_emerg_individuo['nome_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_6, axis=1)
    
    def funcao_cod_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return '2222'
        elif emergency_loc == '6':
            return '2222'
        else:
            return row['cod_espaco_6']
    
    df_emerg_individuo['cod_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_6, axis=1)
    
    def funcao_hr_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['hr_saida_emerg_6']
        elif emergency_loc == '6':
            return row['hr_saida_emerg_6']
        else:
            return row['hr_entrada_7']
    
    df_emerg_individuo['hr_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_7, axis=1)
    
    def funcao_vlc_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_7']
    
    df_emerg_individuo['vlc_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_7, axis=1)
    
    def funcao_hr_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_7'])
        else:
            return row['hr_saida_7']
    
    df_emerg_individuo['hr_saida_emerg_7'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_7, axis=1)
    
    def funcao_vlc_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_7']
    
    df_emerg_individuo['vlc_saida_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_7, axis=1)
    
    def funcao_nome_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'Outros Espaços'
        elif emergency_loc == '6':
            return 'Outros Espaços'
        elif emergency_loc == '7':
            return 'Refeitório'
        else:
            return row['nome_espaco_7']
    
    df_emerg_individuo['nome_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_7, axis=1)
    
    def funcao_cod_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return '1111'
        elif emergency_loc == '6':
            return '1111'
        elif emergency_loc == '7':
            return '3333'
        else:
            return row['cod_espaco_7']
    
    df_emerg_individuo['cod_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_7, axis=1)
    
    def funcao_hr_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['hr_saida_emerg_7']
        else:
            return row['hr_entrada_8']
    
    df_emerg_individuo['hr_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_8, axis=1)
    
    def funcao_vlc_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_8']
    
    df_emerg_individuo['vlc_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_8, axis=1)
    
    def funcao_hr_saida_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_8'])
        else:
            return row['hr_saida_8']
    
    df_emerg_individuo['hr_saida_emerg_8'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_8, axis=1)
    
    def funcao_vlc_saida_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_8']
    
    df_emerg_individuo['vlc_saida_emerg_8'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_8, axis=1)
    
    def funcao_nome_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'Outros Espaços'
        elif emergency_loc == '8':
            return 'Cozinha'
        else:
            return row['nome_espaco_8']
    
    df_emerg_individuo['nome_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_8, axis=1)
    
    def funcao_cod_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return '1111'
        elif emergency_loc == '8':
            return '2222'
        else:
            return row['cod_espaco_8']
    
    df_emerg_individuo['cod_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_8, axis=1)
    
    def funcao_hr_entrada_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return row['hr_saida_emerg_8']
        else:
            return row['hr_entrada_9']
    
    df_emerg_individuo['hr_entrada_emerg_9'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_9, axis=1)
    
    def funcao_vlc_entrada_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return row['vlc_loc_emerg']    
        else:
            return row['vlc_entrada_9']
    
    df_emerg_individuo['vlc_entrada_emerg_9'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_9, axis=1)
    
    def funcao_hr_saida_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_9'])
        else:
            return row['hr_saida_9']
    
    df_emerg_individuo['hr_saida_emerg_9'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_9, axis=1)
    
    def funcao_vlc_saida_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return row['vlc_loc_emerg']  
        else:
            return row['vlc_saida_9']  
    
    df_emerg_individuo['vlc_saida_emerg_9'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_9, axis=1)
    
    def funcao_nome_espaco_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'N/A'
        elif emergency_loc == '8':
            return 'Outros Espaços'
        elif emergency_loc == '9':
            return 'Vestiário/Banheiro'
        else:
            return row['nome_espaco_9']
    
    df_emerg_individuo['nome_espaco_emerg_9'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_9, axis=1)
    
    def funcao_cod_espaco_emerg_9(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'N/A'
        elif emergency_loc == '8':
            return '1111'
        elif emergency_loc == '9':
            return '5555'
        else:
            return row['cod_espaco_9']
    
    df_emerg_individuo['cod_espaco_emerg_9'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_9, axis=1)
    
    def funcao_hr_entrada_emerg_10(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return row['hr_saida_emerg_9']
        else:
            return row['hr_entrada_10']
    
    df_emerg_individuo['hr_entrada_emerg_10'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_10, axis=1)
    
    def funcao_vlc_entrada_emerg_10(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return row['vlc_loc_emerg']     
        else:
            return row['vlc_entrada_10']
    
    df_emerg_individuo['vlc_entrada_emerg_10'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_10, axis=1)
    
    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_10'] = df_emerg_individuo['hr_entrada_emerg_10'] + pd.to_timedelta(additional_times, unit='H')
    
    def funcao_hr_saida_emerg_10(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return np.minimum(row['hr_saida_emerg_10'], row['hr_saida_10'])
        elif emergency_loc == '10':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_10'])
        else:
            return row['hr_saida_10']
    
    df_emerg_individuo['hr_saida_emerg_10'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_10, axis=1)
    
    def funcao_vlc_saida_emerg_10(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return row['vlc_loc_emerg']
        elif emergency_loc == '10':
            return row['vlc_loc_emerg']              
        else:
            return row['vlc_saida_10']  
    
    df_emerg_individuo['vlc_saida_emerg_10'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_10, axis=1)
    
    def funcao_nome_espaco_emerg_10(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'N/A'
        elif emergency_loc == '8':
            return 'N/A'
        elif emergency_loc == '9':
            return 'Cozinha'
        elif emergency_loc == '10':
            return 'Cozinha'
        else:
            return row['nome_espaco_10']
    
    df_emerg_individuo['nome_espaco_emerg_10'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_10, axis=1)
    
    def funcao_cod_espaco_emerg_10(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'N/A'
        elif emergency_loc == '8':
            return 'N/A'
        elif emergency_loc == '9':
            return '2222'
        elif emergency_loc == '10':
            return '2222'
        else:
            return row['cod_espaco_10']
    
    df_emerg_individuo['cod_espaco_emerg_10'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_10, axis=1)
    
    def funcao_hr_entrada_emerg_11(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return row['hr_saida_emerg_10']
        elif emergency_loc == '10':
            return row['hr_saida_emerg_10']
        else:
            return row['hr_entrada_11']
    
    df_emerg_individuo['hr_entrada_emerg_11'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_11, axis=1)
    
    def funcao_vlc_entrada_emerg_11(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.nan
        elif emergency_loc == '7':
            return np.nan
        elif emergency_loc == '8':
            return np.nan
        elif emergency_loc == '9':
            return row['vlc_loc_emerg']
        elif emergency_loc == '10':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_11']
    
    df_emerg_individuo['vlc_entrada_emerg_11'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_11, axis=1)
    
    def funcao_nome_espaco_emerg_11(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'N/A'
        elif emergency_loc == '8':
            return 'N/A'
        elif emergency_loc == '9':
            return 'Outros Espaços'
        elif emergency_loc == '10':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_11']
    
    df_emerg_individuo['nome_espaco_emerg_11'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_11, axis=1)
    
    def funcao_cod_espaco_emerg_11(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'N/A'
        elif emergency_loc == '7':
            return 'N/A'
        elif emergency_loc == '8':
            return 'N/A'
        elif emergency_loc == '9':
            return '1111'
        elif emergency_loc == '10':
            return '1111'
        else:
            return row['cod_espaco_11']
    
    df_emerg_individuo['cod_espaco_emerg_11'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_11, axis=1)
    
    for j in range(1, 11):
        df_emerg_individuo[f'hr_saida_{j}'] = df_emerg_individuo[f'hr_saida_emerg_{j}']
        df_emerg_individuo[f'vlc_saida_{j}'] = df_emerg_individuo[f'vlc_saida_emerg_{j}']
    
    for j in range(2, 12):
        df_emerg_individuo[f'hr_entrada_{j}'] = df_emerg_individuo[f'hr_entrada_emerg_{j}']
        df_emerg_individuo[f'vlc_entrada_{j}'] = df_emerg_individuo[f'vlc_entrada_emerg_{j}']
    
    for j in range(1, 12):
        df_emerg_individuo[f'nome_espaco_{j}'] = df_emerg_individuo[f'nome_espaco_emerg_{j}']
        df_emerg_individuo[f'cod_espaco_{j}'] = df_emerg_individuo[f'cod_espaco_emerg_{j}']
    
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[0:3], axis=1)
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[65:], axis=1)
    
    #Criando comportamento em dia de doenca para cada individuo (tipo = 3000)
     
    sick_days_individuo = random.sample(list(business_days), 2)
    
    df_sick_days_individuo = pd.DataFrame(sick_days_individuo, columns=['data'])
    
    df_sick_days_individuo['cod_individuo'] = i
    df_sick_days_individuo['tipo_individuo'] = 3000
    
    for j in range(1, 11):
        df_sick_days_individuo[f'hr_saida_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_saida_{j}'] = np.nan
    
    for j in range(2, 12):
        df_sick_days_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_entrada_{j}'] = np.nan
    
    for j in range(1, 12):
        df_sick_days_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_sick_days_individuo[f'cod_espaco_{j}'] = 'N/A'
    
    #Criando comportamento de ferias para cada individuo (tipo = 3000)
    
    def sortear_ferias(dias_uteis, ano):
        dias_uteis_ano = dias_uteis[dias_uteis.year == ano]
    
        midyear_candidates = dias_uteis_ano[
            ((dias_uteis_ano.month == 7) | (dias_uteis_ano.month == 8)) & #julho ou agosto
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
    
        endyear_candidates = dias_uteis_ano[
            (dias_uteis_ano.month == 12) | (dias_uteis_ano.month == 1) & #dezembro ou janeiro
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
     
        midyear = random.choice(midyear_candidates)
        endyear = random.choice(endyear_candidates)
    
        return midyear, endyear
    
    ferias = []
    for ano in range(2020, 2024):
        ferias.append(sortear_ferias(business_days, ano))
    
    df_ferias = pd.DataFrame(ferias, columns=['inicio_ferias_meioano', 'inicio_ferias_finalano'])
    
    df_ferias_empilhado = pd.melt(df_ferias, value_name='inicio_ferias')[['inicio_ferias']]
    
    df_ferias_empilhado = df_ferias_empilhado.sort_values(by='inicio_ferias').reset_index(drop=True)
    
    df_ferias_empilhado['termino_ferias'] = df_ferias_empilhado['inicio_ferias'] + pd.Timedelta(days=14)
    
    def get_business_days_between(start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        dias_ut = date_range[date_range.isin(business_days)]
        return dias_ut
    
    df_ferias_empilhado['data'] = df_ferias_empilhado.apply(
        lambda row: get_business_days_between(row['inicio_ferias'], row['termino_ferias']),
        axis=1
        )
    
    df_periodo_ferias_1 = df_ferias_empilhado.explode('data').reset_index(drop=True)
     
    df_ferias_individuo = df_periodo_ferias_1[['data']].dropna().reset_index(drop=True)
    
    df_ferias_individuo.columns = ['data']
     
    df_ferias_individuo = df_ferias_individuo.drop_duplicates()
    
    df_ferias_individuo['cod_individuo'] = i
    df_ferias_individuo['tipo_individuo'] = 3000
    
    for j in range(1, 11):
        df_ferias_individuo[f'hr_saida_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_saida_{j}'] = np.nan
    
    for j in range(2, 12):
        df_ferias_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_entrada_{j}'] = np.nan
    
    for j in range(1, 12):
        df_ferias_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_ferias_individuo[f'cod_espaco_{j}'] = 'N/A'
     
    #Juntando todos os comportamentos para cada individuo (tipo = 3000)
    
    emergencia_datas = df_emerg_individuo['data']
    df_individuo_semerg = df_individuo[~df_individuo['data'].isin(emergencia_datas)]
    df_individuo_emerg = pd.concat([df_individuo_semerg, df_emerg_individuo], ignore_index=True)
    
    sick_days_datas = df_sick_days_individuo['data']
    df_individuo_emerg_sdoenca = df_individuo_emerg[~df_individuo_emerg['data'].isin(sick_days_datas)]
    df_individuo_emerg_doenca = pd.concat([df_individuo_emerg_sdoenca, df_sick_days_individuo], ignore_index=True)
    
    ferias_datas = df_ferias_individuo['data']
    df_individuo_emerg_doenca_sferias = df_individuo_emerg_doenca[~df_individuo_emerg_doenca['data'].isin(ferias_datas)]
    df_ind_final = pd.concat([df_individuo_emerg_doenca_sferias, df_ferias_individuo], ignore_index=True)
    
    # Criando variavel 'ind_emergencia'
    df_ind_final['ind_emergencia'] = df_ind_final['data'].isin(df_emergencia['emergencia']).astype(int)
    
    # Criando variavel 'sit_rh'
    df_ind_final['sit_rh'] = 1 # 1 = (em serviço)
    df_ind_final.loc[df_ind_final['data'].isin(df_sick_days_individuo['data']), 'sit_rh'] = 2 # 2 = (doente -> altera 'em serviço')
    df_ind_final.loc[(df_ind_final['data'].isin(df_ferias_individuo['data'])), 'sit_rh'] = 3 # 3 = (ferias -> altera 'em serviço' e 'doente')
    
    # Ordenando o DataFrame
    df_ind_final = df_ind_final.sort_values(by='data').reset_index(drop=True)
    
    # Salvando o DataFrame no dicionário
    df_t3000 = pd.concat([df_t3000, df_ind_final], ignore_index=True)

#limpando a memoria
#################################################################################################

# Listando todos os objetos na memória
objetos_a_manter = ['business_days', 'df_emergencia', 'df_t1000', 'df_t2000', 'df_t3000', 'objetos_a_manter', 'pd', 'holidays', 'np', 'random', 'sm', 'LogisticRegression', 'cross_val_score', 'KFold', 'accuracy_score']

# Removendo todos os objetos exceto os que estão na lista de objetos a manter
for obj in dir():
    if obj not in objetos_a_manter and not obj.startswith("__"):
        del globals()[obj]
    
# Removendo as variáveis 'objetos_a_manter' e 'obj'
del objetos_a_manter
del obj

#Criando 5 individuos do tipo 4000 (todos os comportamentos possiveis)
#################################################################################
    
# Dataframe para armazenar os DataFrames de cada indivíduo (tipo = 4000)
df_t4000 = pd.DataFrame()
    
# Loop para gerar dados para cada indivíduo de 2001 a 2015
for i in range(4001, 4006):
    
    # Criando o comportamento de rotina para cada individuo (tipo = 4000)
    
    df_individuo = pd.DataFrame(business_days, columns=['data'])
    df_individuo['cod_individuo'] = i
    df_individuo['tipo_individuo'] = 4000
    
    start_time = 16.0  
    end_time = 16.5   
    start_time_in_days = start_time / 24
    end_time_in_days = end_time / 24
    random_times = np.random.uniform(start_time_in_days, end_time_in_days, df_individuo.shape[0])
    df_individuo['hr_saida_1'] = pd.to_datetime(df_individuo['data']) + pd.to_timedelta(random_times, unit='D')
    df_individuo['vlc_saida_1'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_1'] = 'Outros Espaços'
    df_individuo['cod_espaco_1'] = '1111'
        
    df_individuo['hr_entrada_2'] = df_individuo['hr_saida_1']
    df_individuo['vlc_entrada_2'] = df_individuo['vlc_saida_1']
    min_additional_time = 30 / 60
    max_additional_time = 60 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_2'] = df_individuo['hr_entrada_2'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_2'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_2'] = 'Cozinha'
    df_individuo['cod_espaco_2'] = '2222'
        
    df_individuo['hr_entrada_3'] = df_individuo['hr_saida_2']
    df_individuo['vlc_entrada_3'] = df_individuo['vlc_saida_2']
    min_additional_time = 30 / 60
    max_additional_time = 60 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_3'] = df_individuo['hr_entrada_3'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_3'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_3'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_3'] = '5555'
        
    df_individuo['hr_entrada_4'] = df_individuo['hr_saida_3']
    df_individuo['vlc_entrada_4'] = df_individuo['vlc_saida_3']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_4'] = df_individuo['hr_entrada_4'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_4'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_4'] = 'Cozinha'
    df_individuo['cod_espaco_4'] = '2222'
    
    df_individuo['hr_entrada_5'] = df_individuo['hr_saida_4']
    df_individuo['vlc_entrada_5'] = df_individuo['vlc_saida_4']
    min_additional_time = 30 / 60
    max_additional_time = 60 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_5'] = df_individuo['hr_entrada_5'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_5'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_5'] = 'Refeitório'
    df_individuo['cod_espaco_5'] = '3333'
        
    df_individuo['hr_entrada_6'] = df_individuo['hr_saida_5']
    df_individuo['vlc_entrada_6'] = df_individuo['vlc_saida_5']
    min_additional_time = 20 / 60
    max_additional_time = 45 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_6'] = df_individuo['hr_entrada_6'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_6'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_6'] = 'Banheiro'
    df_individuo['cod_espaco_6'] = '4444'
    
    df_individuo['hr_entrada_7'] = df_individuo['hr_saida_6']
    df_individuo['vlc_entrada_7'] = df_individuo['vlc_saida_6']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_7'] = df_individuo['hr_entrada_7'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_7'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_7'] = 'Refeitório'
    df_individuo['cod_espaco_7'] = '3333'
    
    df_individuo['hr_entrada_8'] = df_individuo['hr_saida_7']
    df_individuo['vlc_entrada_8'] = df_individuo['vlc_saida_7']
    df_individuo['nome_espaco_8'] = 'Outros Espaços'
    df_individuo['cod_espaco_8'] = '1111'
    
    # Criando comportamento de emergência para cada individuo (tipo = 4000)
        
    df_emerg_individuo = pd.merge(df_emergencia, df_individuo, left_on='emergencia', right_on='data', how='inner')
        
    def get_location_at_emergency(row):
        emergency_time = row['hr_inicio_emergencia']
        if emergency_time >= row['hr_entrada_8']:
            return '8'
        elif emergency_time >= row['hr_entrada_7']:
            return '7'
        elif emergency_time >= row['hr_entrada_6']:
            return '6'
        elif emergency_time >= row['hr_entrada_5']:
            return '5'
        elif emergency_time >= row['hr_entrada_4']:
            return '4'
        elif emergency_time >= row['hr_entrada_3']:
            return '3'
        elif emergency_time >= row['hr_entrada_2']:
            return '2'
        else:
            return '1'
    
    df_emerg_individuo['loc_mom_emerg'] = df_emerg_individuo.apply(get_location_at_emergency, axis=1)
        
    min_additional_time = 0 / 60
    max_additional_time = 5 / 60
    atraso_reacao_emerg = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['atraso_reacao_emerg'] = pd.to_timedelta(atraso_reacao_emerg, unit='H')
    df_emerg_individuo['vlc_loc_emerg'] = np.random.uniform(9, 12, df_emerg_individuo.shape[0])
            
    def funcao_hr_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_saida_1']
            
    df_emerg_individuo['hr_saida_emerg_1'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_1, axis=1)
            
    def funcao_vlc_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_saida_1']
                
    df_emerg_individuo['vlc_saida_emerg_1'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_1, axis=1)
    
    def funcao_nome_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_1']
    
    df_emerg_individuo['nome_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_1, axis=1)
    
    def funcao_cod_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_1']
    
    df_emerg_individuo['cod_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_1, axis=1)
        
    def funcao_hr_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_entrada_2']
            
    df_emerg_individuo['hr_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_2, axis=1)
            
    def funcao_vlc_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_entrada_2']
                
    df_emerg_individuo['vlc_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_2, axis=1)
                
    def funcao_hr_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_2'])
        else:
            return row['hr_saida_2']
    
    df_emerg_individuo['hr_saida_emerg_2'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_2, axis=1)
    
    def funcao_vlc_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_2']
    
    df_emerg_individuo['vlc_saida_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_2, axis=1)
    
    def funcao_nome_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_2']
    
    df_emerg_individuo['nome_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_2, axis=1)
    
    def funcao_cod_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_2']
    
    df_emerg_individuo['cod_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_2, axis=1)
    
    def funcao_hr_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['hr_saida_emerg_2']
        else:
            return row['hr_entrada_3']
    
    df_emerg_individuo['hr_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_3, axis=1)
    
    def funcao_vlc_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_3']
    
    df_emerg_individuo['vlc_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_3, axis=1)
    
    def funcao_hr_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_3'])
        else:
            return row['hr_saida_3']
    
    df_emerg_individuo['hr_saida_emerg_3'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_3, axis=1)
    
    def funcao_vlc_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_3']
    
    df_emerg_individuo['vlc_saida_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_3, axis=1)
    
    def funcao_nome_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_3']
    
    df_emerg_individuo['nome_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_3, axis=1)
    
    def funcao_cod_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return '1111'
        else:
            return row['cod_espaco_3']
    
    df_emerg_individuo['cod_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_3, axis=1)
    
    def funcao_hr_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_3']
        else:
            return row['hr_entrada_4']
    
    df_emerg_individuo['hr_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_4, axis=1)
    
    def funcao_vlc_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_4']
    
    df_emerg_individuo['vlc_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_4, axis=1)
    
    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo['hr_entrada_emerg_4'] + pd.to_timedelta(additional_times, unit='H')
    
    def funcao_hr_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_saida_emerg_4'], row['hr_saida_4'])
        elif emergency_loc == '4':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_4'])
        else:
            return row['hr_saida_4']
    
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_4, axis=1)
    
    def funcao_vlc_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc =='4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_4']
    
    df_emerg_individuo['vlc_saida_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_4, axis=1)
    
    def funcao_nome_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['nome_espaco_4']
    
    df_emerg_individuo['nome_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_4, axis=1)
    
    def funcao_cod_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['cod_espaco_4']
    
    df_emerg_individuo['cod_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_4, axis=1)
    
    def funcao_hr_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_4']
        elif emergency_loc == '4':
            return row['hr_saida_emerg_4']
        else:
            return row['hr_entrada_5']
    
    df_emerg_individuo['hr_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_5, axis=1)
    
    def funcao_vlc_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc == '4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_5']
    
    df_emerg_individuo['vlc_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_5, axis=1)
    
    def funcao_hr_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_5'])
        else:
            return row['hr_saida_5']
    
    df_emerg_individuo['hr_saida_emerg_5'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_5, axis=1)
    
    def funcao_vlc_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_5']
    
    df_emerg_individuo['vlc_saida_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_5, axis=1)
    
    def funcao_nome_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'Outros Espaços'
        elif emergency_loc == '4':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_5']
    
    df_emerg_individuo['nome_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_5, axis=1)
    
    def funcao_cod_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return '1111'
        elif emergency_loc == '4':
            return '1111'
        else:
            return row['cod_espaco_5']
    
    df_emerg_individuo['cod_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_5, axis=1)
    
    def funcao_hr_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['hr_saida_emerg_5']
        else:
            return row['hr_entrada_6']
    
    df_emerg_individuo['hr_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_6, axis=1)
    
    def funcao_vlc_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_6']
    
    df_emerg_individuo['vlc_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_6, axis=1)
    
    def funcao_hr_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_6'])
        else:
            return row['hr_saida_6']
    
    df_emerg_individuo['hr_saida_emerg_6'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_6, axis=1)
    
    def funcao_vlc_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_6']
    
    df_emerg_individuo['vlc_saida_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_6, axis=1)
    
    def funcao_nome_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'Outros Espaços'
        elif emergency_loc == '6':
            return row['nome_espaco_6']
        else:
            return row['nome_espaco_6']
    
    df_emerg_individuo['nome_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_6, axis=1)
    
    def funcao_cod_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return '1111'
        elif emergency_loc == '6':
            return row['cod_espaco_6']
        else:
            return row['cod_espaco_6']
    
    df_emerg_individuo['cod_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_6, axis=1)
    
    def funcao_hr_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['hr_saida_emerg_6']
        else:
            return row['hr_entrada_7']
    
    df_emerg_individuo['hr_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_7, axis=1)
    
    def funcao_vlc_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_7']
    
    df_emerg_individuo['vlc_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_7, axis=1)
    
    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_7'] = df_emerg_individuo['hr_entrada_emerg_7'] + pd.to_timedelta(additional_times, unit='H')
    
    def funcao_hr_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.minimum(row['hr_saida_emerg_7'], row['hr_saida_7'])
        elif emergency_loc == '7':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_7'])
        else:
            return row['hr_saida_7']
    
    df_emerg_individuo['hr_saida_emerg_7'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_7, axis=1)
    
    def funcao_vlc_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_7']
    
    df_emerg_individuo['vlc_saida_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_7, axis=1)
    
    def funcao_nome_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'Refeitório'
        elif emergency_loc == '7':
            return row['nome_espaco_7']
        else:
            return row['nome_espaco_7']
    
    df_emerg_individuo['nome_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_7, axis=1)
    
    def funcao_cod_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return '3333'
        elif emergency_loc == '7':
            return row['cod_espaco_7']
        else:
            return row['cod_espaco_7']
    
    df_emerg_individuo['cod_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_7, axis=1)
    
    def funcao_hr_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['hr_saida_emerg_7']
        elif emergency_loc == '7':
            return row['hr_saida_emerg_7']
        else:
            return row['hr_entrada_8']
    
    df_emerg_individuo['hr_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_8, axis=1)
    
    def funcao_vlc_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_8']
    
    df_emerg_individuo['vlc_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_8, axis=1)
    
    def funcao_nome_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'Outros Espaços'
        elif emergency_loc == '7':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_8']
    
    df_emerg_individuo['nome_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_8, axis=1)
    
    def funcao_cod_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return '1111'
        elif emergency_loc == '7':
            return '1111'
        else:
            return row['cod_espaco_8']
    
    df_emerg_individuo['cod_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_8, axis=1)
    
    for j in range(1, 8):
        df_emerg_individuo[f'hr_saida_{j}'] = df_emerg_individuo[f'hr_saida_emerg_{j}']
        df_emerg_individuo[f'vlc_saida_{j}'] = df_emerg_individuo[f'vlc_saida_emerg_{j}']
    
    for j in range(2, 9):
        df_emerg_individuo[f'hr_entrada_{j}'] = df_emerg_individuo[f'hr_entrada_emerg_{j}']
        df_emerg_individuo[f'vlc_entrada_{j}'] = df_emerg_individuo[f'vlc_entrada_emerg_{j}']
    
    for j in range(1, 9):
        df_emerg_individuo[f'nome_espaco_{j}'] = df_emerg_individuo[f'nome_espaco_emerg_{j}']
        df_emerg_individuo[f'cod_espaco_{j}'] = df_emerg_individuo[f'cod_espaco_emerg_{j}']
    
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[0:3], axis=1)
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[47:], axis=1)
    
    #Criando comportamento em dia de doenca para cada individuo (tipo = 4000)
     
    sick_days_individuo = random.sample(list(business_days), 2)
    
    df_sick_days_individuo = pd.DataFrame(sick_days_individuo, columns=['data'])
    
    df_sick_days_individuo['cod_individuo'] = i
    df_sick_days_individuo['tipo_individuo'] = 4000
    
    for j in range(1, 8):
        df_sick_days_individuo[f'hr_saida_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_saida_{j}'] = np.nan
    
    for j in range(2, 9):
        df_sick_days_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_entrada_{j}'] = np.nan
    
    for j in range(1, 9):
        df_sick_days_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_sick_days_individuo[f'cod_espaco_{j}'] = 'N/A'
    
    #Criando comportamento de ferias para cada individuo (tipo = 4000)
    
    def sortear_ferias(dias_uteis, ano):
        dias_uteis_ano = dias_uteis[dias_uteis.year == ano]
    
        midyear_candidates = dias_uteis_ano[
            ((dias_uteis_ano.month == 7) | (dias_uteis_ano.month == 8)) & #julho ou agosto
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
    
        endyear_candidates = dias_uteis_ano[
            (dias_uteis_ano.month == 12) | (dias_uteis_ano.month == 1) & #dezembro ou janeiro
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
     
        midyear = random.choice(midyear_candidates)
        endyear = random.choice(endyear_candidates)
    
        return midyear, endyear
    
    ferias = []
    for ano in range(2020, 2024):
        ferias.append(sortear_ferias(business_days, ano))
    
    df_ferias = pd.DataFrame(ferias, columns=['inicio_ferias_meioano', 'inicio_ferias_finalano'])
    
    df_ferias_empilhado = pd.melt(df_ferias, value_name='inicio_ferias')[['inicio_ferias']]
    
    df_ferias_empilhado = df_ferias_empilhado.sort_values(by='inicio_ferias').reset_index(drop=True)
    
    df_ferias_empilhado['termino_ferias'] = df_ferias_empilhado['inicio_ferias'] + pd.Timedelta(days=14)
    
    def get_business_days_between(start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        dias_ut = date_range[date_range.isin(business_days)]
        return dias_ut
    
    df_ferias_empilhado['data'] = df_ferias_empilhado.apply(
        lambda row: get_business_days_between(row['inicio_ferias'], row['termino_ferias']),
        axis=1
        )
    
    df_periodo_ferias_1 = df_ferias_empilhado.explode('data').reset_index(drop=True)
     
    df_ferias_individuo = df_periodo_ferias_1[['data']].dropna().reset_index(drop=True)
    
    df_ferias_individuo.columns = ['data']
     
    df_ferias_individuo = df_ferias_individuo.drop_duplicates()
    
    df_ferias_individuo['cod_individuo'] = i
    df_ferias_individuo['tipo_individuo'] = 4000
    
    for j in range(1, 8):
        df_ferias_individuo[f'hr_saida_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_saida_{j}'] = np.nan
    
    for j in range(2, 9):
        df_ferias_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_entrada_{j}'] = np.nan
    
    for j in range(1, 9):
        df_ferias_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_ferias_individuo[f'cod_espaco_{j}'] = 'N/A'
     
    #Juntando todos os comportamentos para cada individuo (tipo = 4000)
    
    emergencia_datas = df_emerg_individuo['data']
    df_individuo_semerg = df_individuo[~df_individuo['data'].isin(emergencia_datas)]
    df_individuo_emerg = pd.concat([df_individuo_semerg, df_emerg_individuo], ignore_index=True)
    
    sick_days_datas = df_sick_days_individuo['data']
    df_individuo_emerg_sdoenca = df_individuo_emerg[~df_individuo_emerg['data'].isin(sick_days_datas)]
    df_individuo_emerg_doenca = pd.concat([df_individuo_emerg_sdoenca, df_sick_days_individuo], ignore_index=True)
    
    ferias_datas = df_ferias_individuo['data']
    df_individuo_emerg_doenca_sferias = df_individuo_emerg_doenca[~df_individuo_emerg_doenca['data'].isin(ferias_datas)]
    df_ind_final = pd.concat([df_individuo_emerg_doenca_sferias, df_ferias_individuo], ignore_index=True)
    
    # Criando variavel 'ind_emergencia'
    df_ind_final['ind_emergencia'] = df_ind_final['data'].isin(df_emergencia['emergencia']).astype(int)
    
    # Criando variavel 'sit_rh'
    df_ind_final['sit_rh'] = 1 # 1 = (em serviço)
    df_ind_final.loc[df_ind_final['data'].isin(df_sick_days_individuo['data']), 'sit_rh'] = 2 # 2 = (doente -> altera 'em serviço')
    df_ind_final.loc[(df_ind_final['data'].isin(df_ferias_individuo['data'])), 'sit_rh'] = 3 # 3 = (ferias -> altera 'em serviço' e 'doente')
    
    # Ordenando o DataFrame
    df_ind_final = df_ind_final.sort_values(by='data').reset_index(drop=True)
    
    # Salvando o DataFrame no dicionário
    df_t4000 = pd.concat([df_t4000, df_ind_final], ignore_index=True)

#limpando a memoria
#################################################################################################

# Listando todos os objetos na memória
objetos_a_manter = ['business_days', 'df_emergencia', 'df_t1000', 'df_t2000', 'df_t3000', 'df_t4000', 'objetos_a_manter', 'pd', 'holidays', 'np', 'random', 'sm', 'LogisticRegression', 'cross_val_score', 'KFold', 'accuracy_score']

# Removendo todos os objetos exceto os que estão na lista de objetos a manter
for obj in dir():
    if obj not in objetos_a_manter and not obj.startswith("__"):
        del globals()[obj]
    
# Removendo as variáveis 'objetos_a_manter' e 'obj'
del objetos_a_manter
del obj

#Criando 2 individuos do tipo 5000 (todos os comportamentos possiveis)
#################################################################################
    
# Dataframe para armazenar os DataFrames de cada indivíduo (tipo = 5000)
df_t5000 = pd.DataFrame()
    
# Loop para gerar dados para cada indivíduo de 2001 a 2015
for i in range(5001, 5003):
    
    # Criando o comportamento de rotina para cada individuo (tipo = 5000)
    
    df_individuo = pd.DataFrame(business_days, columns=['data'])
    df_individuo['cod_individuo'] = i
    df_individuo['tipo_individuo'] = 5000
    
    start_time = 11.0  
    end_time = 11.5   
    start_time_in_days = start_time / 24
    end_time_in_days = end_time / 24
    random_times = np.random.uniform(start_time_in_days, end_time_in_days, df_individuo.shape[0])
    df_individuo['hr_saida_1'] = pd.to_datetime(df_individuo['data']) + pd.to_timedelta(random_times, unit='D')
    df_individuo['vlc_saida_1'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_1'] = 'Outros Espaços'
    df_individuo['cod_espaco_1'] = '1111'
        
    df_individuo['hr_entrada_2'] = df_individuo['hr_saida_1']
    df_individuo['vlc_entrada_2'] = df_individuo['vlc_saida_1']
    min_additional_time = 30 / 60
    max_additional_time = 45 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_2'] = df_individuo['hr_entrada_2'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_2'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_2'] = 'Cozinha'
    df_individuo['cod_espaco_2'] = '2222'
        
    df_individuo['hr_entrada_3'] = df_individuo['hr_saida_2']
    df_individuo['vlc_entrada_3'] = df_individuo['vlc_saida_2']
    min_additional_time = 15 / 60
    max_additional_time = 25 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_3'] = df_individuo['hr_entrada_3'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_3'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_3'] = 'Vestiário/Banheiro'
    df_individuo['cod_espaco_3'] = '5555'
        
    df_individuo['hr_entrada_4'] = df_individuo['hr_saida_3']
    df_individuo['vlc_entrada_4'] = df_individuo['vlc_saida_3']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_4'] = df_individuo['hr_entrada_4'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_4'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_4'] = 'Cozinha'
    df_individuo['cod_espaco_4'] = '2222'
    
    df_individuo['hr_entrada_5'] = df_individuo['hr_saida_4']
    df_individuo['vlc_entrada_5'] = df_individuo['vlc_saida_4']
    min_additional_time = 30 / 60
    max_additional_time = 45 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_5'] = df_individuo['hr_entrada_5'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_5'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_5'] = 'Refeitório'
    df_individuo['cod_espaco_5'] = '3333'
        
    df_individuo['hr_entrada_6'] = df_individuo['hr_saida_5']
    df_individuo['vlc_entrada_6'] = df_individuo['vlc_saida_5']
    min_additional_time = 5 / 60
    max_additional_time = 15 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_6'] = df_individuo['hr_entrada_6'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_6'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_6'] = 'Banheiro'
    df_individuo['cod_espaco_6'] = '4444'
    
    df_individuo['hr_entrada_7'] = df_individuo['hr_saida_6']
    df_individuo['vlc_entrada_7'] = df_individuo['vlc_saida_6']
    min_additional_time = 2 / 60
    max_additional_time = 5 / 60
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_individuo.shape[0])
    df_individuo['hr_saida_7'] = df_individuo['hr_entrada_7'] + pd.to_timedelta(additional_times, unit='H')
    df_individuo['vlc_saida_7'] = np.random.uniform(4, 6, df_individuo.shape[0])
    df_individuo['nome_espaco_7'] = 'Refeitório'
    df_individuo['cod_espaco_7'] = '3333'
    
    df_individuo['hr_entrada_8'] = df_individuo['hr_saida_7']
    df_individuo['vlc_entrada_8'] = df_individuo['vlc_saida_7']
    df_individuo['nome_espaco_8'] = 'Outros Espaços'
    df_individuo['cod_espaco_8'] = '1111'
    
    # Criando comportamento de emergência para cada individuo (tipo = 5000)
        
    df_emerg_individuo = pd.merge(df_emergencia, df_individuo, left_on='emergencia', right_on='data', how='inner')
        
    def get_location_at_emergency(row):
        emergency_time = row['hr_inicio_emergencia']
        if emergency_time >= row['hr_entrada_8']:
            return '8'
        elif emergency_time >= row['hr_entrada_7']:
            return '7'
        elif emergency_time >= row['hr_entrada_6']:
            return '6'
        elif emergency_time >= row['hr_entrada_5']:
            return '5'
        elif emergency_time >= row['hr_entrada_4']:
            return '4'
        elif emergency_time >= row['hr_entrada_3']:
            return '3'
        elif emergency_time >= row['hr_entrada_2']:
            return '2'
        else:
            return '1'
    
    df_emerg_individuo['loc_mom_emerg'] = df_emerg_individuo.apply(get_location_at_emergency, axis=1)
        
    min_additional_time = 0 / 60
    max_additional_time = 5 / 60
    atraso_reacao_emerg = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['atraso_reacao_emerg'] = pd.to_timedelta(atraso_reacao_emerg, unit='H')
    df_emerg_individuo['vlc_loc_emerg'] = np.random.uniform(9, 12, df_emerg_individuo.shape[0])
            
    def funcao_hr_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_saida_1']
            
    df_emerg_individuo['hr_saida_emerg_1'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_1, axis=1)
            
    def funcao_vlc_saida_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_saida_1']
                
    df_emerg_individuo['vlc_saida_emerg_1'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_1, axis=1)
    
    def funcao_nome_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_1']
    
    df_emerg_individuo['nome_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_1, axis=1)
    
    def funcao_cod_espaco_emerg_1(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_1']
    
    df_emerg_individuo['cod_espaco_emerg_1'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_1, axis=1)
        
    def funcao_hr_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['hr_entrada_2']
            
    df_emerg_individuo['hr_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_2, axis=1)
            
    def funcao_vlc_entrada_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        else:
            return row['vlc_entrada_2']
                
    df_emerg_individuo['vlc_entrada_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_2, axis=1)
                
    def funcao_hr_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_2'])
        else:
            return row['hr_saida_2']
    
    df_emerg_individuo['hr_saida_emerg_2'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_2, axis=1)
    
    def funcao_vlc_saida_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_2']
    
    df_emerg_individuo['vlc_saida_emerg_2'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_2, axis=1)
    
    def funcao_nome_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['nome_espaco_2']
    
    df_emerg_individuo['nome_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_2, axis=1)
    
    def funcao_cod_espaco_emerg_2(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        else:
            return row['cod_espaco_2']
    
    df_emerg_individuo['cod_espaco_emerg_2'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_2, axis=1)
    
    def funcao_hr_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['hr_saida_emerg_2']
        else:
            return row['hr_entrada_3']
    
    df_emerg_individuo['hr_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_3, axis=1)
    
    def funcao_vlc_entrada_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_3']
    
    df_emerg_individuo['vlc_entrada_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_3, axis=1)
    
    def funcao_hr_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_3'])
        else:
            return row['hr_saida_3']
    
    df_emerg_individuo['hr_saida_emerg_3'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_3, axis=1)
    
    def funcao_vlc_saida_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_3']
    
    df_emerg_individuo['vlc_saida_emerg_3'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_3, axis=1)
    
    def funcao_nome_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_3']
    
    df_emerg_individuo['nome_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_3, axis=1)
    
    def funcao_cod_espaco_emerg_3(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return '1111'
        else:
            return row['cod_espaco_3']
    
    df_emerg_individuo['cod_espaco_emerg_3'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_3, axis=1)
    
    def funcao_hr_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_3']
        else:
            return row['hr_entrada_4']
    
    df_emerg_individuo['hr_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_4, axis=1)
    
    def funcao_vlc_entrada_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_4']
    
    df_emerg_individuo['vlc_entrada_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_4, axis=1)
    
    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo['hr_entrada_emerg_4'] + pd.to_timedelta(additional_times, unit='H')
    
    def funcao_hr_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.minimum(row['hr_saida_emerg_4'], row['hr_saida_4'])
        elif emergency_loc == '4':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_4'])
        else:
            return row['hr_saida_4']
    
    df_emerg_individuo['hr_saida_emerg_4'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_4, axis=1)
    
    def funcao_vlc_saida_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc =='4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_4']
    
    df_emerg_individuo['vlc_saida_emerg_4'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_4, axis=1)
    
    def funcao_nome_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['nome_espaco_4']
    
    df_emerg_individuo['nome_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_4, axis=1)
    
    def funcao_cod_espaco_emerg_4(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        else:
            return row['cod_espaco_4']
    
    df_emerg_individuo['cod_espaco_emerg_4'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_4, axis=1)
    
    def funcao_hr_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['hr_saida_emerg_4']
        elif emergency_loc == '4':
            return row['hr_saida_emerg_4']
        else:
            return row['hr_entrada_5']
    
    df_emerg_individuo['hr_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_5, axis=1)
    
    def funcao_vlc_entrada_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return row['vlc_loc_emerg']
        elif emergency_loc == '4':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_5']
    
    df_emerg_individuo['vlc_entrada_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_5, axis=1)
    
    def funcao_hr_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_5'])
        else:
            return row['hr_saida_5']
    
    df_emerg_individuo['hr_saida_emerg_5'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_5, axis=1)
    
    def funcao_vlc_saida_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_5']
    
    df_emerg_individuo['vlc_saida_emerg_5'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_5, axis=1)
    
    def funcao_nome_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'Outros Espaços'
        elif emergency_loc == '4':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_5']
    
    df_emerg_individuo['nome_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_5, axis=1)
    
    def funcao_cod_espaco_emerg_5(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return '1111'
        elif emergency_loc == '4':
            return '1111'
        else:
            return row['cod_espaco_5']
    
    df_emerg_individuo['cod_espaco_emerg_5'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_5, axis=1)
    
    def funcao_hr_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['hr_saida_emerg_5']
        else:
            return row['hr_entrada_6']
    
    df_emerg_individuo['hr_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_6, axis=1)
    
    def funcao_vlc_entrada_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_6']
    
    df_emerg_individuo['vlc_entrada_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_6, axis=1)
    
    def funcao_hr_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_6'])
        else:
            return row['hr_saida_6']
    
    df_emerg_individuo['hr_saida_emerg_6'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_6, axis=1)
    
    def funcao_vlc_saida_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_6']
    
    df_emerg_individuo['vlc_saida_emerg_6'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_6, axis=1)
    
    def funcao_nome_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'Outros Espaços'
        elif emergency_loc == '6':
            return row['nome_espaco_6']
        else:
            return row['nome_espaco_6']
    
    df_emerg_individuo['nome_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_6, axis=1)
    
    def funcao_cod_espaco_emerg_6(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return '1111'
        elif emergency_loc == '6':
            return row['cod_espaco_6']
        else:
            return row['cod_espaco_6']
    
    df_emerg_individuo['cod_espaco_emerg_6'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_6, axis=1)
    
    def funcao_hr_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['hr_saida_emerg_6']
        else:
            return row['hr_entrada_7']
    
    df_emerg_individuo['hr_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_7, axis=1)
    
    def funcao_vlc_entrada_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_7']
    
    df_emerg_individuo['vlc_entrada_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_7, axis=1)
    
    min_additional_time = 1 / 60    # 1 minuto em horas
    max_additional_time = 2 / 60   # 2 minutos em horas
    additional_times = np.random.uniform(min_additional_time, max_additional_time, df_emerg_individuo.shape[0])
    df_emerg_individuo['hr_saida_emerg_7'] = df_emerg_individuo['hr_entrada_emerg_7'] + pd.to_timedelta(additional_times, unit='H')
    
    def funcao_hr_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return np.minimum(row['hr_saida_emerg_7'], row['hr_saida_7'])
        elif emergency_loc == '7':
            return np.minimum(row['hr_inicio_emergencia'] + row['atraso_reacao_emerg'], row['hr_saida_7'])
        else:
            return row['hr_saida_7']
    
    df_emerg_individuo['hr_saida_emerg_7'] = df_emerg_individuo.apply(funcao_hr_saida_emerg_7, axis=1)
    
    def funcao_vlc_saida_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc =='4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_saida_7']
    
    df_emerg_individuo['vlc_saida_emerg_7'] = df_emerg_individuo.apply(funcao_vlc_saida_emerg_7, axis=1)
    
    def funcao_nome_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'Refeitório'
        elif emergency_loc == '7':
            return row['nome_espaco_7']
        else:
            return row['nome_espaco_7']
    
    df_emerg_individuo['nome_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_7, axis=1)
    
    def funcao_cod_espaco_emerg_7(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return '3333'
        elif emergency_loc == '7':
            return row['cod_espaco_7']
        else:
            return row['cod_espaco_7']
    
    df_emerg_individuo['cod_espaco_emerg_7'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_7, axis=1)
    
    def funcao_hr_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['hr_saida_emerg_7']
        elif emergency_loc == '7':
            return row['hr_saida_emerg_7']
        else:
            return row['hr_entrada_8']
    
    df_emerg_individuo['hr_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_hr_entrada_emerg_8, axis=1)
    
    def funcao_vlc_entrada_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return np.nan
        elif emergency_loc == '2':
            return np.nan
        elif emergency_loc == '3':
            return np.nan
        elif emergency_loc == '4':
            return np.nan
        elif emergency_loc == '5':
            return np.nan
        elif emergency_loc == '6':
            return row['vlc_loc_emerg']
        elif emergency_loc == '7':
            return row['vlc_loc_emerg']
        else:
            return row['vlc_entrada_8']
    
    df_emerg_individuo['vlc_entrada_emerg_8'] = df_emerg_individuo.apply(funcao_vlc_entrada_emerg_8, axis=1)
    
    def funcao_nome_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return 'Outros Espaços'
        elif emergency_loc == '7':
            return 'Outros Espaços'
        else:
            return row['nome_espaco_8']
    
    df_emerg_individuo['nome_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_nome_espaco_emerg_8, axis=1)
    
    def funcao_cod_espaco_emerg_8(row):
        emergency_loc = row['loc_mom_emerg']
        if emergency_loc == '1' :
            return 'N/A'
        elif emergency_loc == '2':
            return 'N/A'
        elif emergency_loc == '3':
            return 'N/A'
        elif emergency_loc == '4':
            return 'N/A'
        elif emergency_loc == '5':
            return 'N/A'
        elif emergency_loc == '6':
            return '1111'
        elif emergency_loc == '7':
            return '1111'
        else:
            return row['cod_espaco_8']
    
    df_emerg_individuo['cod_espaco_emerg_8'] = df_emerg_individuo.apply(funcao_cod_espaco_emerg_8, axis=1)
    
    for j in range(1, 8):
        df_emerg_individuo[f'hr_saida_{j}'] = df_emerg_individuo[f'hr_saida_emerg_{j}']
        df_emerg_individuo[f'vlc_saida_{j}'] = df_emerg_individuo[f'vlc_saida_emerg_{j}']
    
    for j in range(2, 9):
        df_emerg_individuo[f'hr_entrada_{j}'] = df_emerg_individuo[f'hr_entrada_emerg_{j}']
        df_emerg_individuo[f'vlc_entrada_{j}'] = df_emerg_individuo[f'vlc_entrada_emerg_{j}']
    
    for j in range(1, 9):
        df_emerg_individuo[f'nome_espaco_{j}'] = df_emerg_individuo[f'nome_espaco_emerg_{j}']
        df_emerg_individuo[f'cod_espaco_{j}'] = df_emerg_individuo[f'cod_espaco_emerg_{j}']
    
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[0:3], axis=1)
    df_emerg_individuo = df_emerg_individuo.drop(df_emerg_individuo.columns[47:], axis=1)
    
    #Criando comportamento em dia de doenca para cada individuo (tipo = 5000)
     
    sick_days_individuo = random.sample(list(business_days), 2)
    
    df_sick_days_individuo = pd.DataFrame(sick_days_individuo, columns=['data'])
    
    df_sick_days_individuo['cod_individuo'] = i
    df_sick_days_individuo['tipo_individuo'] = 5000
    
    for j in range(1, 8):
        df_sick_days_individuo[f'hr_saida_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_saida_{j}'] = np.nan
    
    for j in range(2, 9):
        df_sick_days_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_sick_days_individuo[f'vlc_entrada_{j}'] = np.nan
    
    for j in range(1, 9):
        df_sick_days_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_sick_days_individuo[f'cod_espaco_{j}'] = 'N/A'
    
    #Criando comportamento de ferias para cada individuo (tipo = 5000)
    
    def sortear_ferias(dias_uteis, ano):
        dias_uteis_ano = dias_uteis[dias_uteis.year == ano]
    
        midyear_candidates = dias_uteis_ano[
            ((dias_uteis_ano.month == 7) | (dias_uteis_ano.month == 8)) & #julho ou agosto
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
    
        endyear_candidates = dias_uteis_ano[
            (dias_uteis_ano.month == 12) | (dias_uteis_ano.month == 1) & #dezembro ou janeiro
            (dias_uteis_ano.weekday <= 2)  # 0=segunda, 1=terça, 2=quarta
            ]
     
        midyear = random.choice(midyear_candidates)
        endyear = random.choice(endyear_candidates)
    
        return midyear, endyear
    
    ferias = []
    for ano in range(2020, 2024):
        ferias.append(sortear_ferias(business_days, ano))
    
    df_ferias = pd.DataFrame(ferias, columns=['inicio_ferias_meioano', 'inicio_ferias_finalano'])
    
    df_ferias_empilhado = pd.melt(df_ferias, value_name='inicio_ferias')[['inicio_ferias']]
    
    df_ferias_empilhado = df_ferias_empilhado.sort_values(by='inicio_ferias').reset_index(drop=True)
    
    df_ferias_empilhado['termino_ferias'] = df_ferias_empilhado['inicio_ferias'] + pd.Timedelta(days=14)
    
    def get_business_days_between(start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        dias_ut = date_range[date_range.isin(business_days)]
        return dias_ut
    
    df_ferias_empilhado['data'] = df_ferias_empilhado.apply(
        lambda row: get_business_days_between(row['inicio_ferias'], row['termino_ferias']),
        axis=1
        )
    
    df_periodo_ferias_1 = df_ferias_empilhado.explode('data').reset_index(drop=True)
     
    df_ferias_individuo = df_periodo_ferias_1[['data']].dropna().reset_index(drop=True)
    
    df_ferias_individuo.columns = ['data']
     
    df_ferias_individuo = df_ferias_individuo.drop_duplicates()
    
    df_ferias_individuo['cod_individuo'] = i
    df_ferias_individuo['tipo_individuo'] = 5000
    
    for j in range(1, 8):
        df_ferias_individuo[f'hr_saida_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_saida_{j}'] = np.nan
    
    for j in range(2, 9):
        df_ferias_individuo[f'hr_entrada_{j}'] = pd.NaT
        df_ferias_individuo[f'vlc_entrada_{j}'] = np.nan
    
    for j in range(1, 9):
        df_ferias_individuo[f'nome_espaco_{j}'] = 'N/A'
        df_ferias_individuo[f'cod_espaco_{j}'] = 'N/A'
     
    #Juntando todos os comportamentos para cada individuo (tipo = 5000)
    
    emergencia_datas = df_emerg_individuo['data']
    df_individuo_semerg = df_individuo[~df_individuo['data'].isin(emergencia_datas)]
    df_individuo_emerg = pd.concat([df_individuo_semerg, df_emerg_individuo], ignore_index=True)
    
    sick_days_datas = df_sick_days_individuo['data']
    df_individuo_emerg_sdoenca = df_individuo_emerg[~df_individuo_emerg['data'].isin(sick_days_datas)]
    df_individuo_emerg_doenca = pd.concat([df_individuo_emerg_sdoenca, df_sick_days_individuo], ignore_index=True)
    
    ferias_datas = df_ferias_individuo['data']
    df_individuo_emerg_doenca_sferias = df_individuo_emerg_doenca[~df_individuo_emerg_doenca['data'].isin(ferias_datas)]
    df_ind_final = pd.concat([df_individuo_emerg_doenca_sferias, df_ferias_individuo], ignore_index=True)
    
    # Criando variavel 'ind_emergencia'
    df_ind_final['ind_emergencia'] = df_ind_final['data'].isin(df_emergencia['emergencia']).astype(int)
    
    # Criando variavel 'sit_rh'
    df_ind_final['sit_rh'] = 1 # 1 = (em serviço)
    df_ind_final.loc[df_ind_final['data'].isin(df_sick_days_individuo['data']), 'sit_rh'] = 2 # 2 = (doente -> altera 'em serviço')
    df_ind_final.loc[(df_ind_final['data'].isin(df_ferias_individuo['data'])), 'sit_rh'] = 3 # 3 = (ferias -> altera 'em serviço' e 'doente')
    
    # Ordenando o DataFrame
    df_ind_final = df_ind_final.sort_values(by='data').reset_index(drop=True)
    
    # Salvando o DataFrame no dicionário
    df_t5000 = pd.concat([df_t5000, df_ind_final], ignore_index=True)

#limpando a memoria
#################################################################################################

# Listando todos os objetos na memória
objetos_a_manter = ['business_days', 'df_emergencia', 'df_t1000', 'df_t2000', 'df_t3000', 'df_t4000', 'df_t5000', 'objetos_a_manter', 'pd', 'holidays', 'np', 'random', 'sm', 'LogisticRegression', 'cross_val_score', 'KFold', 'accuracy_score']

# Removendo todos os objetos exceto os que estão na lista de objetos a manter
for obj in dir():
    if obj not in objetos_a_manter and not obj.startswith("__"):
        del globals()[obj]
    
# Removendo as variáveis 'objetos_a_manter' e 'obj'
del objetos_a_manter
del obj

#empilhando todos os tipos de individuo
################################################################################################

df_final = pd.concat([df_t1000, df_t2000, df_t3000, df_t4000, df_t5000], join='outer')

del([df_t1000, df_t2000, df_t3000, df_t4000, df_t5000])

###############################################################################################
#Modelo de previsao ###########################################################################
###############################################################################################

#substituindo valores nao observados
#######################################################################################

random_offsets = np.random.uniform(-120, 120, size=df_final.shape[0])
random_times = pd.Timestamp('01:00:00') + pd.to_timedelta(random_offsets, unit='s')

for j in range(1, 11): # Substituir pd.NaT por 1 hora da manhã (mais ou menos 2 minutos)
    df_final[f'hr_saida_{j}'] = np.where(df_final[f'hr_saida_{j}'].isna(), random_times, df_final[f'hr_saida_{j}'])

random_offsets = np.random.uniform(-120, 120, size=df_final.shape[0])
random_times = pd.Timestamp('01:00:00') + pd.to_timedelta(random_offsets, unit='s')

for j in range(2, 12): # Substituir pd.NaT por 1 hora da manhã (mais ou menos 2 minutos)
    df_final[f'hr_entrada_{j}'] = np.where(df_final[f'hr_entrada_{j}'].isna(), random_times, df_final[f'hr_entrada_{j}'])

random_speeds = np.random.uniform(0.9, 1.1, size=df_final.shape[0])

for j in range(1, 11): # Substituir NaN por 1 (mais ou menos 0,1)
    df_final[f'vlc_saida_{j}'] = np.where(df_final[f'vlc_saida_{j}'].isna(), random_speeds, df_final[f'vlc_saida_{j}'])

random_speeds = np.random.uniform(0.9, 1.1, size=df_final.shape[0])

for j in range(2, 12): # Substituir NaN por 1 (mais ou menos 0,1)
    df_final[f'vlc_entrada_{j}'] = np.where(df_final[f'vlc_entrada_{j}'].isna(), random_speeds, df_final[f'vlc_entrada_{j}'])

for j in range(1, 12):
    df_final[f'cod_espaco_{j}'] = df_final[f'cod_espaco_{j}'].fillna('9999') # substitui NaN por '9999'
    df_final[f'cod_espaco_{j}'] = df_final[f'cod_espaco_{j}'].replace('N/A', '9999') # substitui 'N/A' por '9999'

del([j, random_offsets, random_times, random_speeds])

#convertendo as variaveis de interesse em numericas
##########################################################################################

for j in range(1, 11):
    df_final[f'hr_saida_num_{j}'] = df_final[f'hr_saida_{j}'].dt.hour + df_final[f'hr_saida_{j}'].dt.minute / 60 + df_final[f'hr_saida_{j}'].dt.second / 3600

for j in range(2, 12):
    df_final[f'hr_entrada_num_{j}'] = df_final[f'hr_entrada_{j}'].dt.hour + df_final[f'hr_entrada_{j}'].dt.minute / 60 + df_final[f'hr_entrada_{j}'].dt.second / 3600

for j in range(1, 12):
    df_final[f'cod_espaco_num_{j}'] = df_final[f'cod_espaco_{j}'].astype(int)

del([j])

#convertendo df_final para long (inves de wide): acho que isso vai ajudar nas regressoes
##########################################################################################

hr_saida_columns = [f'hr_saida_num_{i}' for i in range(1, 11)]
df_hrsaida = pd.melt(df_final, id_vars=['data', 'cod_individuo'], value_vars=hr_saida_columns, var_name='esp_numero', value_name='hr_saida')
df_hrsaida['esp_numero'] = df_hrsaida['esp_numero'].str.extract('(\d+)').astype(int)

hr_entrada_columns = [f'hr_entrada_num_{i}' for i in range(2, 12)]
df_hrentrada = pd.melt(df_final, id_vars=['data', 'cod_individuo'], value_vars=hr_entrada_columns, var_name='esp_numero', value_name='hr_entrada')
df_hrentrada['esp_numero'] = df_hrentrada['esp_numero'].str.extract('(\d+)').astype(int)

vlc_saida_columns = [f'vlc_saida_{i}' for i in range(1, 11)]
df_vlcsaida = pd.melt(df_final, id_vars=['data', 'cod_individuo'], value_vars=vlc_saida_columns, var_name='esp_numero', value_name='vlc_saida')
df_vlcsaida['esp_numero'] = df_vlcsaida['esp_numero'].str.extract('(\d+)').astype(int)

vlc_entrada_columns = [f'vlc_entrada_{i}' for i in range(2, 12)]
df_vlcentrada = pd.melt(df_final, id_vars=['data', 'cod_individuo'], value_vars=vlc_entrada_columns, var_name='esp_numero', value_name='vlc_entrada')
df_vlcentrada['esp_numero'] = df_vlcentrada['esp_numero'].str.extract('(\d+)').astype(int)

cod_espaco_columns = [f'cod_espaco_num_{i}' for i in range(1, 12)]
df_codespaco = pd.melt(df_final, id_vars=['data', 'tipo_individuo', 'cod_individuo', 'sit_rh', 'ind_emergencia'], value_vars=cod_espaco_columns, var_name='esp_numero', value_name='cod_espaco')
df_codespaco['esp_numero'] = df_codespaco['esp_numero'].str.extract('(\d+)').astype(int)

del([hr_entrada_columns, hr_saida_columns, vlc_entrada_columns, vlc_saida_columns, cod_espaco_columns])

df_reg1 = pd.merge(df_codespaco, df_hrsaida,
                     on=['data', 'cod_individuo', 'esp_numero'], 
                     how='outer')

df_reg2 = pd.merge(df_reg1, df_hrentrada, 
                     on=['data', 'cod_individuo', 'esp_numero'], 
                     how='outer')

df_reg3 = pd.merge(df_reg2, df_vlcsaida, 
                     on=['data', 'cod_individuo', 'esp_numero'], 
                     how='outer')

df_reg_final = pd.merge(df_reg3, df_vlcentrada, 
                     on=['data', 'cod_individuo', 'esp_numero'], 
                     how='outer')

del([df_final, df_codespaco, df_hrsaida, df_hrentrada, df_vlcsaida, df_vlcentrada, df_reg1, df_reg2, df_reg3])

#criando as variaveis do modelo
##########################################################################################

#padronizando horas de saida

media_desvio = df_reg_final.groupby(['tipo_individuo', 'sit_rh', 'esp_numero'])['hr_saida'].agg(['mean', 'std']).reset_index()
media_desvio.rename(columns={'mean': 'hr_saida_media', 'std': 'hr_saida_std'}, inplace=True)
df_reg_final = df_reg_final.merge(media_desvio, on=['tipo_individuo', 'sit_rh', 'esp_numero'], how='left')

df_reg_final['hr_saida_pad'] = np.where((df_reg_final['hr_saida_std'] == 0) | (np.isnan(df_reg_final['hr_saida_std'])), 
                                             0, 
                                             (df_reg_final['hr_saida'] - df_reg_final['hr_saida_media']) / df_reg_final['hr_saida_std'])

del([media_desvio])

#padronizando horas de entrada

media_desvio = df_reg_final.groupby(['tipo_individuo', 'sit_rh', 'esp_numero'])['hr_entrada'].agg(['mean', 'std']).reset_index()
media_desvio.rename(columns={'mean': 'hr_entrada_media', 'std': 'hr_entrada_std'}, inplace=True)
df_reg_final = df_reg_final.merge(media_desvio, on=['tipo_individuo', 'sit_rh', 'esp_numero'], how='left')

df_reg_final['hr_entrada_pad'] = np.where((df_reg_final['hr_entrada_std'] == 0) | (np.isnan(df_reg_final['hr_entrada_std'])), 
                                             0, 
                                             (df_reg_final['hr_entrada'] - df_reg_final['hr_entrada_media']) / df_reg_final['hr_entrada_std'])

del([media_desvio])

#padronizando velocidade de saida

media_desvio = df_reg_final.groupby(['tipo_individuo', 'sit_rh', 'esp_numero'])['vlc_saida'].agg(['mean', 'std']).reset_index()
media_desvio.rename(columns={'mean': 'vlc_saida_media', 'std': 'vlc_saida_std'}, inplace=True)
df_reg_final = df_reg_final.merge(media_desvio, on=['tipo_individuo', 'sit_rh', 'esp_numero'], how='left')

df_reg_final['vlc_saida_pad'] = np.where((df_reg_final['vlc_saida_std'] == 0) | (np.isnan(df_reg_final['vlc_saida_std'])), 
                                             0, 
                                             (df_reg_final['vlc_saida'] - df_reg_final['vlc_saida_media']) / df_reg_final['vlc_saida_std'])

del([media_desvio])

#padronizando velocidade de entrada

media_desvio = df_reg_final.groupby(['tipo_individuo', 'sit_rh', 'esp_numero'])['vlc_entrada'].agg(['mean', 'std']).reset_index()
media_desvio.rename(columns={'mean': 'vlc_entrada_media', 'std': 'vlc_entrada_std'}, inplace=True)
df_reg_final = df_reg_final.merge(media_desvio, on=['tipo_individuo', 'sit_rh', 'esp_numero'], how='left')

df_reg_final['vlc_entrada_pad'] = np.where((df_reg_final['vlc_entrada_std'] == 0) | (np.isnan(df_reg_final['vlc_entrada_std'])), 
                                             0, 
                                             (df_reg_final['vlc_entrada'] - df_reg_final['vlc_entrada_media']) / df_reg_final['vlc_entrada_std'])

del([media_desvio])

#padronizando codigo do espaco

moda_espaco = df_reg_final.groupby(['tipo_individuo', 'sit_rh', 'esp_numero'])['cod_espaco'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
moda_espaco.rename(columns={'cod_espaco': 'cod_espaco_moda'}, inplace=True)
ruido = np.random.choice([0, 1], size=len(moda_espaco), p=[0.8, 0.2])
moda_espaco['cod_espaco_moda'] += ruido #introduzindo ruido em moda_espaco senão terei problemas nos meus modelos
df_reg_final = df_reg_final.merge(moda_espaco, on=['tipo_individuo', 'sit_rh', 'esp_numero'], how='left')
df_reg_final['cod_espaco_ver'] = np.where(df_reg_final['cod_espaco'] != df_reg_final['cod_espaco_moda'], 1, 0)

del([moda_espaco])

'''
moda_espaco = df_reg_final.groupby(['tipo_individuo', 'sit_rh', 'esp_numero'])['cod_espaco'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
moda_espaco.rename(columns={'cod_espaco': 'cod_espaco_moda'}, inplace=True)
df_reg_final = df_reg_final.merge(moda_espaco, on=['tipo_individuo', 'sit_rh', 'esp_numero'], how='left')
df_reg_final['cod_espaco_ver'] = np.where(df_reg_final['cod_espaco'] != df_reg_final['cod_espaco_moda'], 1, 0)

del([moda_espaco])
'''
#criando dummies a partir de "sit_rh"

df_reg_final = pd.get_dummies(df_reg_final, columns=['sit_rh'], prefix='sit_rh', drop_first=False)

df_reg_final['sit_rh_1'] = df_reg_final['sit_rh_1'].replace({True: 1, False: 0})
df_reg_final['sit_rh_2'] = df_reg_final['sit_rh_2'].replace({True: 1, False: 0})
df_reg_final['sit_rh_3'] = df_reg_final['sit_rh_3'].replace({True: 1, False: 0})

#criando dummies a partir de "tipo_individuo"

df_reg_final = pd.get_dummies(df_reg_final, columns=['tipo_individuo'], prefix='tipo_individuo', drop_first=False)

df_reg_final['tipo_individuo_1000'] = df_reg_final['tipo_individuo_1000'].replace({True: 1, False: 0})
df_reg_final['tipo_individuo_2000'] = df_reg_final['tipo_individuo_2000'].replace({True: 1, False: 0})
df_reg_final['tipo_individuo_3000'] = df_reg_final['tipo_individuo_3000'].replace({True: 1, False: 0})
df_reg_final['tipo_individuo_4000'] = df_reg_final['tipo_individuo_4000'].replace({True: 1, False: 0})
df_reg_final['tipo_individuo_5000'] = df_reg_final['tipo_individuo_5000'].replace({True: 1, False: 0})

#jogando fora variaveis que nao vou usar no modelo
#######################################################################################

df_reg_final = df_reg_final.drop(columns=['vlc_saida', 'vlc_saida_media', 'vlc_saida_std', 'hr_saida', 'hr_saida_media', 'hr_saida_std'])

df_reg_final = df_reg_final.drop(columns=['vlc_entrada', 'vlc_entrada_media', 'vlc_entrada_std', 'hr_entrada', 'hr_entrada_media', 'hr_entrada_std'])

df_reg_final = df_reg_final.drop(columns=['cod_espaco', 'cod_espaco_moda'])

#estimando os modelos
#######################################################################################

# Vetor y

y = df_reg_final['ind_emergencia']

# Matrix X

var_exp = []

var_exp.append('hr_saida_pad')
var_exp.append('vlc_saida_pad')
var_exp.append('hr_entrada_pad')
var_exp.append('vlc_entrada_pad')
var_exp.append('cod_espaco_ver')

for i in range(1,3):
    var_exp.append(f'sit_rh_{i}')

for i in [1000, 2000, 3000, 4000]:
    var_exp.append(f'tipo_individuo_{i}')

X = df_reg_final[var_exp]
X = sm.add_constant(X)

del([i])

#MPL

kf = KFold(n_splits=5)
mpl_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    mpl_model = sm.OLS(y_train, X_train).fit(disp=False)
    
    y_pred_prob = mpl_model.predict(X_test)
    
    y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
    
    mpl_accuracy = accuracy_score(y_test, y_pred)
    mpl_scores.append(mpl_accuracy)

print("MPL - Média de acurácia:", np.mean(mpl_scores))
print("MPL - Desvio padrão de acurácia:", np.std(mpl_scores))

#PROBIT

kf = KFold(n_splits=5)
probit_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    probit_model = sm.Probit(y_train, X_train).fit(disp=False)
    
    y_pred_prob = probit_model.predict(X_test)
    y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
    
    probit_accuracy = accuracy_score(y_test, y_pred)
    probit_scores.append(probit_accuracy)

print("Probit - Média de acurácia:", np.mean(probit_scores))
print("Probit - Desvio padrão de acurácia:", np.std(probit_scores))

#LOGIT

logit_model = LogisticRegression()
logit_scores = cross_val_score(logit_model, X, y, cv=5, scoring='accuracy')
print("Logit - Média de acurácia:", np.mean(logit_scores))
print("Logit - Desvio padrão de acurácia:", np.std(logit_scores))

#Modelo logit e o melhor (mas a acuracia e muito parecida para todos os modelos)

