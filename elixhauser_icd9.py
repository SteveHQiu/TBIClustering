import re
import json

def extractICD(text):
    text = text.split("\n")
    icd_codes = []
    for line in text: 
        if re.search(r"when icd9_code = '(\w*)'", line) != None:            
            icd_codes.append(re.search(r"when icd9_code = '(\w*)'", line).group(1))
        if re.search(r"when icd9_code between '(\w*)' and '(\w*)'", line) != None:
            start = re.search(r"when icd9_code between '(\w*)' and '(\w*)'", line).group(1)
            end = re.search(r"when icd9_code between '(\w*)' and '(\w*)'", line).group(2)
            prefix = re.search(r"(\D)\d*", start)
            if prefix == None:
                icd_codes.extend([str(i) for i in list(range(int(start), int(end)+1))]) # Convert all to string 
            else: # ICD code prefixed by a letter 
                start = re.search(r"when icd9_code between '\D(\d*)' and '\D(\d*)'", line).group(1)
                end = re.search(r"when icd9_code between '\D(\d*)' and '\D(\d*)'", line).group(2)
                icd_codes.extend([prefix.group(1) + str(i) for i in list(range(int(start), int(end)+1))]) # Convert all to string 
                prefix.group(1)
    return icd_codes

#%%
cardiac_arrhythmia = """
    when icd9_code = '42610' then 1
    when icd9_code = '42611' then 1
    when icd9_code = '42613' then 1
    when icd9_code between '4262' and '42653' then 1
    when icd9_code between '4266' and '42689' then 1
    when icd9_code = '4270' then 1
    when icd9_code = '4272' then 1
    when icd9_code = '42731' then 1
    when icd9_code = '42760' then 1
    when icd9_code = '4279' then 1
    when icd9_code = '7850' then 1
    when icd9_code between 'V450' and 'V4509' then 1
    when icd9_code between 'V533' and 'V5339' then 1

"""
a = extractICD(cardiac_arrhythmia)
#%%

dictionary = dict()

congestive_heart_failure = """
  when icd9_code = '39891' then 1
  when icd9_code between '4280' and '4289' then 1
  when icd9_code = '40201' then 1
  when icd9_code = '40211' then 1
  when icd9_code = '40291'         then 1
  when icd9_code = '40401' then 1
  when icd9_code = '40411' then 1
  when icd9_code = '40491'         then 1
  when icd9_code = '40403' then 1
  when icd9_code = '40413' then 1
  when icd9_code = '40493'         then 1

"""
dictionary["congestive_heart_failure"] = extractICD(congestive_heart_failure)

 # Uses Enhanced ICD9 Elixhauser
dictionary["cardiac_arrhythmia"] = ["4260", "42610", "42612", "42613", "4267", "4269", "4270", "4271", "4272", "42731", "42732", "42741", "42742", "42760", "42761", "42769", "42781", "42789", "4279", "7850", "99601", "99604", "V4500", "V4501", "V4502", "V4509", "V5331", "V5332", "V5339"] 

valvular_disease = """
when icd9_code between '09320' and '09324' then 1
  when icd9_code between '3940' and '3971' then 1
  when icd9_code = '3979' then 1
  when icd9_code between '4240' and '4243' then 1
  when icd9_code = '42490' then 1
  when icd9_code = '42491' then 1
  when icd9_code = '42499' then 1
  when icd9_code between '7463' and '7466' then 1
  when icd9_code = 'V422' then 1
  when icd9_code = 'V433' then 1

"""
dictionary["valvular_disease"] = extractICD(valvular_disease)

pulmonary_circulation_disorder = """
  when icd9_code between '41511' and '41519' then 1
  when icd9_code between '4160' and '4169' then 1
  when icd9_code = '4179' then 1

"""
dictionary["pulmonary_circulation_disorder"] = extractICD(pulmonary_circulation_disorder)

peripheral_vascular_disorder = """
  when icd9_code between '4400' and '4409' then 1
  when icd9_code between '44100' and '44103' then 1
  when icd9_code between '4411' and '4419' then 1
  when icd9_code between '4420' and '4429' then 1
  when icd9_code between '4431' and '4439' then 1
  when icd9_code between '44421' and '44422' then 1
  when icd9_code = '4471' then 1
  when icd9_code = '449' then 1
  when icd9_code = '5571' then 1
  when icd9_code = '5579' then 1
  when icd9_code = 'V434' then 1
"""

dictionary["peripheral_vascular_disorder"] = extractICD(peripheral_vascular_disorder)

hypertension_uncomplicated = """
  when icd9_code = '4011' then 1
  when icd9_code = '4019' then 1
  when icd9_code between '64200' and '64204' then 1

"""
dictionary["hypertension_uncomplicated"] = extractICD(hypertension_uncomplicated)

hypertension_complicated = """
  when icd9_code = '4010' then 1
  when icd9_code = '4372' then 1
  when icd9_code between '64220' and '64224' then 1
  when icd9_code = '40200' then 1
    when icd9_code = '40210' then 1
    when icd9_code = '40290' then 1
    when icd9_code = '40509' then 1
    when icd9_code = '40519' then 1
    when icd9_code = '40599'         then 1
  when icd9_code = '40300' then 1
  when icd9_code = '40310' then 1
  when icd9_code = '40390' then 1
  when icd9_code = '40501' then 1
  when icd9_code = '40511' then 1
  when icd9_code = '40591' then 1
  when icd9_code between '64210' and '64214' then 1
  when icd9_code = '40301' then 1
  when icd9_code = '40311' then 1
  when icd9_code = '40391'         then 1
  when icd9_code = '40400' then 1
  when icd9_code = '40410' then 1
  when icd9_code = '40490'         then 1
  when icd9_code = '40402' then 1
  when icd9_code = '40412' then 1
  when icd9_code = '40492'         then 1
  when icd9_code = '40401' then 1
  when icd9_code = '40411' then 1
  when icd9_code = '40491'         then 1
  when icd9_code = '40403' then 1
  when icd9_code = '40413' then 1
  when icd9_code = '40493'         then 1
  when icd9_code between '64270' and '64274' then 1
  when icd9_code between '64290' and '64294' then 1


"""
dictionary["hypertension_complicated"] = extractICD(hypertension_complicated)

paralysis = """
  when icd9_code between '3420' and '3449' then 1
  when icd9_code between '43820' and '43853' then 1
  when icd9_code = '78072'         then 1

"""
dictionary["paralysis"] = extractICD(paralysis)

other_neurological_disorder = """
when icd9_code between '3300' and '3319' then 1
  when icd9_code = '3320' then 1
  when icd9_code = '3334' then 1
  when icd9_code = '3335' then 1
  when icd9_code = '3337' then 1
  when icd9_code = '33371' then 1
  when icd9_code = '33372' then 1
  when icd9_code = '33379' then 1
  when icd9_code = '33385' then 1
  when icd9_code = '33394' then 1
  when icd9_code between '3340' and '3359' then 1
  when icd9_code = '3380' then 1
  when icd9_code = '340' then 1
  when icd9_code between '3411' and '3419' then 1
  when icd9_code between '34500' and '34511' then 1
  when icd9_code between '3452' and '3453' then 1
  when icd9_code between '34540' and '34591' then 1
  when icd9_code between '34700' and '34701' then 1
  when icd9_code between '34710' and '34711' then 1
  when icd9_code = '3483' then 1 -- discontinued icd-9
  when icd9_code between '64940' and '64944' then 1
  when icd9_code = '7687' then 1
  when icd9_code between '76870' and '76873' then 1
  when icd9_code = '7803' then 1
  when icd9_code = '78031' then 1
  when icd9_code = '78032' then 1
  when icd9_code = '78033' then 1
  when icd9_code = '78039' then 1
  when icd9_code = '78097' then 1
  when icd9_code = '7843'         then 1

"""
dictionary["other_neurological_disorder"] = extractICD(other_neurological_disorder)

chronic_pulmonary_disease = """
  when icd9_code = '490' then 1
  when icd9_code between '4910' and '4928' then 1
  when icd9_code between '49300' and '49392' then 1
  when icd9_code between '4940' and '4941' then 1
  when icd9_code between '4950' and '4959' then 1
  when icd9_code between '496' and '505' then 1
  when icd9_code = '5064' then 1

"""
dictionary["chronic_pulmonary_disease"] = extractICD(chronic_pulmonary_disease)

diabetes_uncomplicated = """
  when icd9_code between '25000' and '25033' then 1
  when icd9_code between '64800' and '64804' then 1
  when icd9_code between '24900' and '24931' then 1

"""
dictionary["diabetes_uncomplicated"] = extractICD(diabetes_uncomplicated)

diabetes_complicated = """
  when icd9_code between '25040' and '25093' then 1
  when icd9_code = '7751' then 1
  when icd9_code between '24940' and '24991' then 1

"""
dictionary["diabetes_complicated"] = extractICD(diabetes_complicated)

hypothyroidism = """
  when icd9_code = '243' then 1
  when icd9_code between '2440' and '2442' then 1
  when icd9_code = '2448' then 1
  when icd9_code = '2449'         then 1

"""
dictionary["hypothyroidism"] = extractICD(hypothyroidism)

renal_failure = """
  when icd9_code = '585' then 1 -- discontinued code
  when icd9_code = '5853' then 1
  when icd9_code = '5854' then 1
  when icd9_code = '5855' then 1
  when icd9_code = '5856' then 1
  when icd9_code = '5859' then 1
  when icd9_code = '586' then 1
  when icd9_code = 'V420' then 1
  when icd9_code = 'V451' then 1
  when icd9_code between 'V560' and 'V563' then 1
  when icd9_code between 'V5631' and 'V5632' then 1
  when icd9_code = 'V568' then 1
  when icd9_code between 'V4511' and 'V4512' then 1

"""
dictionary["renal_failure"] = extractICD(renal_failure)

liver_disease = """
  when icd9_code = '07022' then 1
  when icd9_code = '07023' then 1
  when icd9_code = '07032' then 1
  when icd9_code = '07033' then 1
  when icd9_code = '07044' then 1
  when icd9_code = '07054' then 1
  when icd9_code = '4560' then 1
  when icd9_code = '4561' then 1
  when icd9_code = '45620' then 1
  when icd9_code = '45621' then 1
  when icd9_code = '5710' then 1
  when icd9_code = '5712' then 1
  when icd9_code = '5713' then 1
  when icd9_code between '57140' and '57149' then 1
  when icd9_code = '5715' then 1
  when icd9_code = '5716' then 1
  when icd9_code = '5718' then 1
  when icd9_code = '5719' then 1
  when icd9_code = '5723' then 1
  when icd9_code = '5728' then 1
  when icd9_code = '5735' then 1
  when icd9_code = 'V427'         then 1

"""
dictionary["liver_disease"] = extractICD(liver_disease)

peptic_ulcer_disease_excluding_bleeding = """
  when icd9_code = '53141' then 1
  when icd9_code = '53151' then 1
  when icd9_code = '53161' then 1
  when icd9_code = '53170' then 1
  when icd9_code = '53171' then 1
  when icd9_code = '53191' then 1
  when icd9_code = '53241' then 1
  when icd9_code = '53251' then 1
  when icd9_code = '53261' then 1
  when icd9_code = '53270' then 1
  when icd9_code = '53271' then 1
  when icd9_code = '53291' then 1
  when icd9_code = '53341' then 1
  when icd9_code = '53351' then 1
  when icd9_code = '53361' then 1
  when icd9_code = '53370' then 1
  when icd9_code = '53371' then 1
  when icd9_code = '53391' then 1
  when icd9_code = '53441' then 1
  when icd9_code = '53451' then 1
  when icd9_code = '53461' then 1
  when icd9_code = '53470' then 1
  when icd9_code = '53471' then 1
  when icd9_code = '53491'         then 1

"""
dictionary["peptic_ulcer_disease_excluding_bleeding"] = extractICD(peptic_ulcer_disease_excluding_bleeding)

aids_hiv = """
  when icd9_code = '42' then 1
  when icd9_code = '7953' then 1
  when icd9_code = 'V08' then 1
"""
dictionary["aids_hiv"] = extractICD(aids_hiv)

lymphoma = """
  when icd9_code between '20000' and '20238' then 1
  when icd9_code between '20250' and '20301' then 1
  when icd9_code = '2386' then 1
  when icd9_code = '2733' then 1
  when icd9_code between '20302' and '20382' then 1

"""
dictionary["lymphoma"] = extractICD(lymphoma)

metastatic_cancer = """
  when icd9_code between '1960' and '1991' then 1
  when icd9_code between '20970' and '20975' then 1
  when icd9_code = '20979' then 1
  when icd9_code = '78951'         then 1

"""
dictionary["metastatic_cancer"] = extractICD(metastatic_cancer)

solid_tumor_wo_metastasis = """
  when icd9_code between '1400' and '1729' then 1
  when icd9_code between '1740' and '1759' then 1
  when icd9_code between '179' and '195' then 1
  when icd9_code between '1790' and '1958' then 1
  when icd9_code between '20900' and '20936' then 1
  when icd9_code between '25801' and '25803' then 1

"""
dictionary["solid_tumor_wo_metastasis"] = extractICD(solid_tumor_wo_metastasis)

rheumatoid_arhritis = """
  when icd9_code = '7010' then 1
  when icd9_code between '7100' and '7109' then 1
  when icd9_code between '7140' and '7149' then 1
  when icd9_code between '7200' and '7209' then 1
  when icd9_code = '725' then 1

"""
dictionary["rheumatoid_arhritis"] = extractICD(rheumatoid_arhritis)

coagulopathy = """
  when icd9_code between '2860' and '2869' then 1
  when icd9_code = '2871' then 1
  when icd9_code between '2873' and '2875' then 1
  when icd9_code between '64930' and '64934' then 1
  when icd9_code = '28984'         then 1

"""
dictionary["coagulopathy"] = extractICD(coagulopathy)

obesity = """
  when icd9_code = '2780' then 1
  when icd9_code = '27800' then 1
  when icd9_code = '27801' then 1
  when icd9_code = '27803' then 1
  when icd9_code between '64910' and '64914' then 1
  when icd9_code between 'V8530' and 'V8539' then 1
  when icd9_code = 'V854' then 1 -- hierarchy used for AHRQ v3.6 and earlier
  when icd9_code between 'V8541' and 'V8545' then 1
  when icd9_code = 'V8554' then 1
  when icd9_code = '79391'         then 1

"""
dictionary["obesity"] = extractICD(obesity)

weight_loss = """
  when icd9_code = '262' then 1
  when icd9_code between '2630' and '2639' then 1
  when icd9_code between '78321' and '78322' then 1

"""
dictionary["weight_loss"] = extractICD(weight_loss)

fluid_and_electrolyte_disorders = """
  when icd9_code between '2760' and '2769' then 1
"""
dictionary["fluid_and_electrolyte_disorders"] = extractICD(fluid_and_electrolyte_disorders)

blood_loss_anemia = """
  when icd9_code = '2800' then 1
  when icd9_code between '64820' and '64824' then 1

"""
dictionary["blood_loss_anemia"] = extractICD(blood_loss_anemia)

deficiency_anemia = """
  when icd9_code between '2801' and '2819' then 1
  when icd9_code between '28521' and '28529' then 1
  when icd9_code = '2859'         then 1

"""
dictionary["deficiency_anemia"] = extractICD(deficiency_anemia)

alcohol_abuse = """
  when icd9_code between '2910' and '2913' then 1
  when icd9_code = '2915' then 1
  when icd9_code = '2918' then 1
  when icd9_code = '29181' then 1
  when icd9_code = '29182' then 1
  when icd9_code = '29189' then 1
  when icd9_code = '2919' then 1
  when icd9_code between '30300' and '30393' then 1
  when icd9_code between '30500' and '30503' then 1

"""
dictionary["alcohol_abuse"] = extractICD(alcohol_abuse)

drug_abuse = """
  when icd9_code = '2920' then 1
  when icd9_code between '29282' and '29289' then 1
  when icd9_code = '2929' then 1
  when icd9_code between '30400' and '30493' then 1
  when icd9_code between '30520' and '30593' then 1
  when icd9_code between '64830' and '64834' then 1

"""
dictionary["drug_abuse"] = extractICD(drug_abuse)

psychoses = """
  when icd9_code between '29500' and '29595' then 1
  when icd9_code between '2971' and '2981' then 1
  when icd9_code between '2984' and '2989' then 1
  when icd9_code = '29910' then 1
  when icd9_code = '29911'         then 1

"""
dictionary["psychoses"] = extractICD(psychoses)

depression = """
  when icd9_code = '3004' then 1
  when icd9_code = '30112' then 1
  when icd9_code = '3090' then 1
  when icd9_code = '3091' then 1
  when icd9_code = '311'         then 1

"""
dictionary["depression"] = extractICD(depression)



#%%

with open("icd9.json", "w") as file: 
    json.dump(dictionary, file)