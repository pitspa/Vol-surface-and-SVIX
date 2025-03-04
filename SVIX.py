import numpy as np
import scipy.stats
import pandas
import matplotlib
import matplotlib.pyplot as mp
from time import time
pandas.options.mode.chained_assignment = None  # default='warn'

#%%

def BScall(K,r,T,S0,sigma):
    d1 = (np.log(S0/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*scipy.stats.norm.cdf(d1) - K*np.exp(-r*T)*scipy.stats.norm.cdf(d2)

def BSput(K,r,T,S0,sigma):
    d1 = (np.log(S0/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return - S0*scipy.stats.norm.cdf(-d1) + K*np.exp(-r*T)*scipy.stats.norm.cdf(-d2)

def static_replication_nth_moment(n,K_array,calls_array,puts_array,Rf,S0):
    
    dK_array = np.diff(K_array)
    
    dK_left_hand = np.where(K_array[1:]<=S0*Rf,dK_array,0) 
    dK_right_hand = np.where(K_array[1:]>S0*Rf,dK_array,0) 
    
    first_integral = np.sum((K_array[1:]/S0-Rf)**(n-2)*puts_array[1:]*dK_left_hand)
    second_integral = np.sum((K_array[1:]/S0-Rf)**(n-2)*calls_array[1:]*dK_right_hand)
    
    return(n*(n-1)*Rf/S0**2)*(first_integral+second_integral)

#%%

df = pandas.read_csv(r"C:\Users\Pietro\Desktop\Misc\Data\SP_500_options_2022_2023_csv\rcudhhntk6ynq1et.csv")
df['date'] = pandas.to_datetime(df['date'])
dates = np.unique(df['date'])

df_index_prices = pandas.read_csv(r"C:\Users\Pietro\Desktop\Misc\Data\SP_500_prices_2022_2023\qqsqvq1xddibgbt5.csv")
df_index_prices['date'] = pandas.to_datetime(df_index_prices['date'])

index_avg_daily_log_return = np.average(np.diff(np.log(df_index_prices['close'])))
index_var_daily_log_return = np.var(np.diff(np.log(df_index_prices['close'])))

#%%

target_horizon = 30 # horizon considered - NOT financial days
highest_maturity_horizon = 365 # upper bound for maturities considered in the surface
lowest_maturity_horizon = 15  # lower bound for maturities considered in the surface

rn_pdfs = []
vol_surfaces = []
SVIX_ = []
modal_pk = []

verbose = 1

for k in range(0,len(dates)):
    
    start_time = time()
    
    date = pandas.to_datetime(str(dates[k]))
    year = date.year
    month = date.month
    day = date.day
    
    # daily options 
    
    table_options = df[(df['date'].dt.year==year)&
               (df['date'].dt.month==month)&
               (df['date'].dt.day==day)]
    
    # daily index price and price at horizon
    
    index_price = np.array(
                df_index_prices[(df_index_prices['date'].dt.year==year)&
               (df_index_prices['date'].dt.month==month)&
               (df_index_prices['date'].dt.day==day)]['close'])[0]
    
    horizon_date = date + pandas.DateOffset(days=target_horizon)
    index_price_horizon = df_index_prices.iloc[(df_index_prices['date']-horizon_date).abs().argsort()[0]]['close']
    
    #
    
    date_strikes = np.array((table_options.loc[:,'strike_price'])/1000).astype('float64')
    date_mid_prices = (np.array(table_options['best_bid']) + np.array(table_options['best_offer']))/2
    
    table_options.loc[:,'renormalised_strikes'] = date_strikes
    table_options.loc[:,'midprice'] = date_mid_prices

    date_maturities = np.unique(table_options['exdate'])
    
    impvols = []
    
    for ell in range(0,len(date_maturities)):
        
        maturity = pandas.to_datetime(date_maturities[ell])
        maturity_horizon = (maturity - date).days # NOT financial days
        
        if np.logical_and(maturity_horizon >= lowest_maturity_horizon,
                             maturity_horizon <= highest_maturity_horizon):
        
            table_options_maturity = table_options[pandas.to_datetime(table_options['exdate']) == maturity]
            table_options_strikes_maturity = np.unique(table_options_maturity['renormalised_strikes'])
            
            for nu in range(0,len(table_options_strikes_maturity)):
                
                strike = table_options_strikes_maturity[nu]
                
                call = np.array(table_options_maturity[(table_options_maturity['cp_flag'] == 'C')&
                                                 (table_options_maturity['renormalised_strikes'] == strike)]['midprice'])
                put = np.array(table_options_maturity[(table_options_maturity['cp_flag'] == 'P')&
                                                 (table_options_maturity['renormalised_strikes'] == strike)]['midprice'])
                
                if np.logical_and(call.size == 1, put.size == 1):
                
                    boxrate_gross = 1/(-(call - put - index_price)/strike)
                    box_instantaneous_rate = np.log(boxrate_gross)/(maturity_horizon/365)
                    
                    # choose OTM contract for the impvol
                    
                    if index_price <= strike:
                        
                        def diff(sigma):
                            return (call - BScall(strike,
                                                 box_instantaneous_rate,
                                                 maturity_horizon/365,
                                                 index_price,
                                                 sigma))
                        
                        impvol = scipy.optimize.brentq(diff, 1e-08, 10)
                        
                    elif index_price >= strike:
                        
                        def diff(sigma):
                            return (put - BSput(table_options_strikes_maturity[nu],
                                                box_instantaneous_rate,
                                                maturity_horizon/365,
                                                index_price,
                                                sigma))
                        
                        impvol = scipy.optimize.brentq(diff, 1e-08, 10)
            
                    impvols.append([date,impvol,strike,maturity_horizon,index_price,boxrate_gross])
                
    impvols = pandas.DataFrame(impvols)
    
    vol_surfaces.append([date,impvols])
    
    maturity_horizons_unique = np.unique(impvols.loc[:,3])
    strikes_unique = np.unique(impvols.loc[:,2])
        
    tck = scipy.interpolate.bisplrep(x=np.array(impvols[2]).astype('float64'),
                                     y=np.array(impvols[3]).astype('float64'),
                                     z=np.array(impvols[1]).astype('float64'),
                                     s=3)
    
    tck_box = scipy.interpolate.bisplrep(x=np.array(impvols[2]).astype('float64'),
                                     y=np.array(impvols[3]).astype('float64'),
                                     z=np.array(impvols[5]).astype('float64'),
                                     s=1)
    
    strikerange_artificial = np.linspace(index_price*0.65,index_price*1.35,1000)
    interpol = scipy.interpolate.bisplev(strikerange_artificial, target_horizon, tck)
    interpol_box = scipy.interpolate.bisplev(strikerange_artificial, target_horizon, tck_box)
    avg_box = np.average(interpol_box)
    
    if verbose == 1:
        
        f, axarr = mp.subplots(1,2,sharex=False,figsize=(12,4.5))
        ax1, ax2 = axarr.flatten()
        
        for h in range(0,len(maturity_horizons_unique)):
            
            impvolplot = impvols[impvols[3]==maturity_horizons_unique[h]]
            
            if np.logical_and(maturity_horizons_unique[h] <= target_horizon + 20,
                                 maturity_horizons_unique[h] >= target_horizon - 20):
                
                ax1.plot(np.array(impvolplot.loc[:,2]).astype('float64'),
                        np.array(impvolplot.loc[:,1]).astype('float64'),
                        '.',
                        label=maturity_horizons_unique[h]
                        )
                
                ax2.plot(np.array(impvolplot.loc[:,2]).astype('float64'),
                        np.array(impvolplot.loc[:,5]).astype('float64'),
                        '.',
                        label=maturity_horizons_unique[h]
                        )
            
        ax1.plot(strikerange_artificial,interpol.T[0],'k',label=target_horizon)
        ax2.plot(strikerange_artificial,interpol_box.T[0],'k',label=target_horizon)
        ax2.plot(strikerange_artificial,np.full(shape=len(strikerange_artificial),
                                                   fill_value=avg_box),'k--',
                 label='avg box rate {} days'.format(target_horizon))

        ax1.set_title(date)
        ax2.set_title(date)
        ax1.set_ylabel('Impvol')
        ax2.set_ylabel('Box rates')
        ax1.legend()
        ax2.legend()
        
        mp.show()
    
    callrange_artificial = np.zeros(len(strikerange_artificial))
    putrange_artificial = np.zeros(len(strikerange_artificial))
    
    for j_ in range(0,len(strikerange_artificial)):
        
        callrange_artificial[j_] = BScall(strikerange_artificial[j_],
                                          np.log(avg_box)/(target_horizon/365),
                                          target_horizon/365,
                                          index_price,
                                          interpol[j_][0])
        
        putrange_artificial[j_] = BSput(strikerange_artificial[j_],
                                          np.log(avg_box)/(target_horizon/365),
                                          target_horizon/365,
                                          index_price,
                                          interpol[j_][0])
        
    # Martin SVIX
    
    MQ_2 = static_replication_nth_moment(n=2,
                                         K_array=strikerange_artificial,
                                         calls_array=callrange_artificial,
                                         puts_array=putrange_artificial,
                                         Rf=avg_box,
                                         S0=index_price)
    
    SVIX_.append([date,MQ_2,avg_box,MQ_2/avg_box])
        
    # Breeden-Litzenberger
    
    Q_cdf = 1 + avg_box*np.gradient(callrange_artificial,strikerange_artificial)
    Q_pdf = np.gradient(Q_cdf,strikerange_artificial)
    
    rn_pdfs.append([date,strikerange_artificial,Q_pdf,index_price])
    
    if verbose == 1:
        
        f, axarr = mp.subplots(1,2,sharex=False,figsize=(12,4.5))
        ax1, ax2 = axarr.flatten()
        
        ax1.axvline(index_price,color='black',label='Current price')
        ax1.plot(strikerange_artificial,Q_cdf,label='RN cdf {} days'.format(target_horizon))
        ax1.set_title(date)
        ax1.set_xlim((strikerange_artificial[0],strikerange_artificial[-1]))
        ax1.legend()
        
        ax2.axvline(index_price,color='black',label='Current price')
        ax2.plot(strikerange_artificial,Q_pdf,label='RN pdf {} days'.format(target_horizon))
        ax2.set_title(date)
        ax2.set_xlim((strikerange_artificial[0],strikerange_artificial[-1]))
        ax2.legend()
        
        mp.show()
    
    end_time = time()
    print('Time elapsed', end_time - start_time)
    
    print(date)
    
    
#%%

SVIX_ = np.array(SVIX_)

dateplot = SVIX_[:,0]
mp.plot(dateplot,np.zeros(len(dateplot)))
mp.plot(dateplot,SVIX_[:,-1],label='SVIX bound')
mp.legend()
mp.show()



