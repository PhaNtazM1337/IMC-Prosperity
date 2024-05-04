import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
from collections import OrderedDict
from collections import deque
from collections import defaultdict
import numpy as np
import copy
import statistics
from statistics import mean, stdev
empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}

def def_value():
    return copy.deepcopy(empty_dict)

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.positions = {}
        self.positions['AMETHYSTS'] = 0
        self.positions['STARFRUIT'] = 0
        self.positions['ORCHIDS'] = 0
        self.positions['ORCHIDS_trade'] = 0
        self.positions['GIFT_BASKET'] = 0
        self.positions['CHOCOLATE'] = 0
        self.positions['STRAWBERRIES'] = 0
        self.positions['ROSES'] = 0
        self.positions['COCONUT'] = 0
        self.positions['COCONUT_COUPON'] = 0
        self.basket_mean = 379.49
        self.basket_std = 76.42
        self.coconut_resid_std = 13.465114
        self.STARFRUIT_window = deque(maxlen=8)
        self.coconut_len = 480
        self.coconut_coeff = 1.6
        self.coconut_window = deque(maxlen = self.coconut_len)
        self.prev_vals = None
        self.ORCHIDS_ma = deque(maxlen=40)
        self.ORCHIDS_slopes = deque(maxlen = 100)
        # params
        self.AMETHYSTS_lb = 10001
        self.AMETHYSTS_ub = 9999
        self.cont_buy_basket_unfill = 0
        self.cont_sell_basket_unfill = 0
        self.mm_order_size = 30
        self.mm_scale_down = 15
        self.epsilon = 0
        self.orchids = 0
        self.dist = statistics.NormalDist(mu=0.0, sigma=1.0)
        self.prev_orchid_conversions = 0
        self.position_limits = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
        self.person_position = defaultdict(def_value)

    def black_scholes(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = self.dist.cdf(d1)
        d2 = d1 - sigma * np.sqrt(T)
        C = S * self.dist.cdf(d1) - K * np.exp(-r * T) * self.dist.cdf(d2)
        return C, delta

    def calc_coupon(self, order_depth):
        orders = {'COCONUT': [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p in prods:
            osell[p] = OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
            if len(osell[p])==0 or len(obuy[p])==0:
                return orders
            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))
            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))
            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol
                if vol_buy[p] >= self.position_limits[p] / 10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol
                if vol_sell[p] >= self.position_limits[p] / 10:
                    break
        mid = mid_price['COCONUT']
        self.coconut_window.append(mid)
        c_pred, delta = self.black_scholes(mid_price['COCONUT'], 10000, 250 / 365.25, 0.0, 0.1923)
        res = mid_price['COCONUT_COUPON'] - c_pred
        trade_at = self.coconut_resid_std * 0.4
        close_at = self.coconut_resid_std * (-0.4)

        # if len(self.coconut_window) == self.coconut_len:
        #     ma = mean(self.coconut_window)
        #     std = stdev(self.coconut_window)
        #     bb_upper, bb_lower = ma + self.coconut_coeff * std, ma - self.coconut_coeff * std
        #     if self.prev_vals == None:
        #         self.prev_vals = (bb_lower, bb_upper, mid)
        #         return orders
        #     else:
        #         prev_bb_lower, prev_bb_upper, prev_mid = self.prev_vals
        #         coco_pos = self.positions['COCONUT']
        #         limit = self.position_limits['COCONUT']
        #         if prev_mid < prev_bb_lower and mid > bb_lower and coco_pos < limit:
        #             #buy
        #             for price, qty in osell['COCONUT'].items():
        #                 if coco_pos < limit:
        #                     order_size = min(-qty, limit - coco_pos)
        #                     coco_pos += order_size
        #                     orders['COCONUT'].append(Order("COCONUT", price, order_size))
        #         if prev_mid > prev_bb_upper and mid < bb_upper and coco_pos > -limit:
        #             #sell
        #             for price, qty in obuy['COCONUT'].items():
        #                 if coco_pos > -limit:
        #                     order_size = max(-qty, -limit - coco_pos)
        #                     coco_pos += order_size
        #                     orders['COCONUT'].append(Order("COCONUT", price, order_size))
        #         self.prev_vals = (bb_lower, bb_upper, mid)


        if res > trade_at:
            vol = self.positions['COCONUT_COUPON'] + self.position_limits['COCONUT_COUPON']
            assert (vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -vol))
        elif res < -trade_at:
            vol = self.position_limits['COCONUT_COUPON'] - self.positions['COCONUT_COUPON']
            assert (vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], vol))
        elif res < close_at and self.positions['COCONUT_COUPON'] < 0:
            vol = -self.positions['COCONUT_COUPON']
            assert (vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], vol))
        elif res > -close_at and self.positions['COCONUT_COUPON'] > 0:
            vol = self.positions['COCONUT_COUPON']
            assert (vol >= 0)
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -vol))

        return orders

    def calc_qty(self, orderlist, sell):
        qty = 0
        max_qty_availble = 0
        opt_ask = -1
        for (i,j) in orderlist:
            if sell:
                 j = -j
            qty += j
            if j > max_qty_availble:
                max_qty_availble = j
                opt_ask = i
        return opt_ask, max_qty_availble


    def calc_AMETHYSTS(self, orderbook):
        orders = []
        asks = sorted(orderbook.sell_orders.items())
        bids = sorted(orderbook.buy_orders.items(), reverse = True)
        pos = self.positions['AMETHYSTS']
        lmt = self.position_limits['AMETHYSTS']
        ask_price, max_ask_qty = self.calc_qty(asks,True)
        bid_price, max_bid_qty = self.calc_qty(bids,False)

        # Undercut within bound
        mm_ask = max(ask_price - 1, self.AMETHYSTS_ub + 1)
        mm_bid = min(bid_price + 1, self.AMETHYSTS_lb - 1)

        # Buy if below 10000
        for price, qty in asks:
            if ((price < self.AMETHYSTS_lb) or ((self.positions['AMETHYSTS']<0) and (price == self.AMETHYSTS_lb))) and pos < lmt:
                order_size = min(-qty, lmt - pos)
                pos += order_size
                orders.append(Order('AMETHYSTS', price, order_size))

        if pos < lmt and self.positions['AMETHYSTS'] < 0:
            order_size = min(self.mm_order_size, lmt - pos)
            orders.append(Order('AMETHYSTS', min(bid_price + 2, self.AMETHYSTS_lb - 1), order_size))
            pos += order_size

        if (pos < lmt) and (self.positions['AMETHYSTS'] > self.mm_scale_down):
            order_size = min(self.mm_order_size, lmt - pos)
            orders.append(Order('AMETHYSTS', min(bid_price, self.AMETHYSTS_lb - 1), order_size))
            pos += order_size

        if pos < lmt:
            order_size = min(self.mm_order_size, lmt - pos)
            orders.append(Order('AMETHYSTS', mm_bid, order_size))
            pos += order_size


        pos = self.positions['AMETHYSTS']

        for price, qty in bids:
            if ((price > self.AMETHYSTS_ub) or ((self.positions['AMETHYSTS']>0) and (price == self.AMETHYSTS_ub))) and pos > -lmt:
                order_size = max(-qty, -lmt - pos)
                pos += order_size
                orders.append(Order('AMETHYSTS', price, order_size))


        if pos > -lmt and self.positions['AMETHYSTS'] > 0:
            order_size = max(-self.mm_order_size, -lmt - pos)
            orders.append(Order('AMETHYSTS', max(ask_price - 2, self.AMETHYSTS_ub + 1), order_size))
            pos += order_size

        if (pos > -lmt) and (self.positions['AMETHYSTS'] < -self.mm_scale_down):
            order_size = max(-self.mm_order_size, -lmt - pos)
            orders.append(Order('AMETHYSTS', max(ask_price, self.AMETHYSTS_ub + 1), order_size))
            pos += order_size

        if pos >-lmt:
            order_size = max(-self.mm_order_size, -lmt - pos)
            orders.append(Order('AMETHYSTS', mm_ask, order_size))
            pos += order_size

        return orders

    def calc_STARFRUIT(self, orderbook, lb, ub):
        orders = []
        asks = sorted(orderbook.sell_orders.items())
        bids = sorted(orderbook.buy_orders.items(), reverse=True)
        pos = self.positions['STARFRUIT']
        lmt = self.position_limits['STARFRUIT']
        ask_price, max_ask_qty = self.calc_qty(asks, True)
        bid_price, max_bid_qty = self.calc_qty(bids, False)
        mm_bid = min(bid_price + 1, lb)
        mm_ask = max(ask_price - 1, ub)

        for price, qty in asks:
            if ((price <= lb) or ((self.positions["STARFRUIT"] < 0) and (price == lb + 1))) and pos < lmt:
                order_size = min(-qty, lmt - pos)
                pos += order_size
                orders.append(Order("STARFRUIT", price, order_size))

        if pos < lmt:
            order_size = lmt - pos
            orders.append(Order("STARFRUIT", mm_bid, order_size))
            pos += order_size

        pos = self.positions["STARFRUIT"]

        for price, qty in bids:
            if ((price>= ub) or ((self.positions["STARFRUIT"] > 0) and (price == ub - 1))) and pos > -lmt:
                order_size = max(-qty, -lmt - pos)
                pos += order_size
                orders.append(Order("STARFRUIT", price, order_size))

        if pos > -lmt:
            order_size = -lmt - pos
            orders.append(Order("STARFRUIT", mm_ask, order_size))
            pos += order_size

        return orders

    def calc_orchids(self, orchid_obs, domestic_mkt):
        lmt = self.position_limits['ORCHIDS']
        # trade_lmt = self.position_limits['ORCHIDS_trade']
        pos = self.positions['ORCHIDS']
        # if self.orchids == -1:
        #     pos -= trade_lmt
        # elif self.orchids == 1:
        #     pos += trade_lmt
        # perform arbitrage if possible
        sorders = []
        borders = []
        sconversion = 0
        bconversion = 0

        ask_price = orchid_obs.askPrice
        bid_price = orchid_obs.bidPrice
        transport_fee = orchid_obs.transportFees
        ex_tariff = orchid_obs.exportTariff
        in_tariff = orchid_obs.importTariff
        eff_buy_price = ask_price + transport_fee + in_tariff
        eff_sell_price = bid_price - transport_fee - ex_tariff

        asks = sorted(domestic_mkt.sell_orders.items())
        bids = sorted(domestic_mkt.buy_orders.items(), reverse=True)
        bpnl = 0
        for price, qty in bids:
            ordered = 0
            if price > eff_buy_price:
                order_size = max(-qty, -lmt - pos)
                borders.append(Order("ORCHIDS", price, order_size))
                pos += order_size
                bconversion += qty
                bpnl += (price - eff_buy_price) * (-order_size)
                ordered = -order_size
            # if sell_signal and qty - ordered >= trade_lmt and self.orchids != -1:
            #     self.orchids = -1
            #     order_size = -trade_lmt
            #     orders.append(Order("ORCHIDS", price, order_size))
            #     print("sold")

        pos = self.positions['ORCHIDS']
        # if self.orchids == -1:
        #     pos -= trade_lmt
        # elif self.orchids == 1:
        #     pos += trade_lmt

        spnl = 0
        for price, qty in asks:
            ordered = 0
            if price < eff_sell_price:
                order_size = min(-qty, lmt-pos)
                sorders.append(Order("ORCHIDS", price, order_size))
                pos += order_size
                spnl -= order_size * 0.1
                spnl += (eff_sell_price - price) * order_size
                sconversion += qty
            # if buy_signal and -qty - ordered >= trade_lmt and self.orchids != 1:
            #     self.orchids = 1
            #     order_size = trade_lmt
            #     orders.append(Order("ORCHIDS", price, order_size))
            #     print("bought")
        if bpnl > spnl:
            return borders, bconversion
        if spnl > bpnl:
            return sorders, sconversion
        return [], 0

        # market making using pred price

    def calc_basket(self, order_depth):

        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol
                if vol_buy[p] >= self.position_limits[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol
                if vol_sell[p] >= self.position_limits[p]/10:
                    break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - self.basket_mean
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - self.basket_mean

        trade_at = self.basket_std*0.25
        close_at = self.basket_std*(-1000)

        pb_pos = self.positions['GIFT_BASKET']
        pb_neg = self.positions['GIFT_BASKET']


        basket_buy_sig = 0
        basket_sell_sig = 0

        if self.positions['GIFT_BASKET'] == self.position_limits['GIFT_BASKET']:
            self.cont_buy_basket_unfill = 0
        if self.positions['GIFT_BASKET'] == -self.position_limits['GIFT_BASKET']:
            self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.positions['GIFT_BASKET'] + self.position_limits['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol))
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
                #uku_pos += vol
        elif res_buy < -trade_at:
            vol = self.position_limits['GIFT_BASKET'] - self.positions['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                self.cont_buy_basket_unfill += 2
                pb_pos += vol

        p = "ROSES"
        if self.person_position["Rhianna"][p] == 1:
            vol = self.position_limits[p] - self.positions[p]
            orders[p].append(Order(p, best_sell[p], vol))
        if self.person_position["Rhianna"][p] == -1:
            vol = self.positions[p] + self.position_limits[p]
            orders[p].append(Order(p, best_buy[p], -vol))

        p = "CHOCOLATE"
        if self.person_position["Vladimir"][p] == 1:
            vol = self.position_limits[p] - self.positions[p]
            orders[p].append(Order(p, best_sell[p], vol))
        if self.person_position["Vladimir"][p] == -1:
            vol = self.positions[p] + self.position_limits[p]
            orders[p].append(Order(p, best_buy[p], -vol))

        return orders

    def pred_price(self):
        const = 0.9532
        weights = [-0.0037, 0.0082, -0.0130, 0.0063, 0.0185, 0.0282, 0.1917, 0.7636]
        price = const
        for i in range(len(self.STARFRUIT_window)):
            price += weights[i]*self.STARFRUIT_window[i]
        return int(round(price))


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        trader_data = ""
        prev_slope = 0

        for product, order_depth in state.order_depths.items():
            orders[product] = []
        #
        for key, val in state.position.items():
            self.positions[key] = val

        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                self.person_position[trade.buyer][product] = 1
                self.person_position[trade.seller][product] = -1


        orders['AMETHYSTS'] += self.calc_AMETHYSTS(state.order_depths['AMETHYSTS'])


        ask_STARFRUIT = self.calc_qty(sorted(state.order_depths['STARFRUIT'].sell_orders.items()), True)
        bid_STARFRUIT = self.calc_qty(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True), False)
        midpx_STARFRUIT = (ask_STARFRUIT[0] + bid_STARFRUIT[0]) / 2
        self.STARFRUIT_window.append(midpx_STARFRUIT)
        next_price = -1
        if len(self.STARFRUIT_window)==8:
            next_price = self.pred_price()
        if next_price>0:
            STARFRUIT_lb = next_price - 1
            STARFRUIT_ub = next_price + 1
            orders['STARFRUIT'] += self.calc_STARFRUIT(state.order_depths['STARFRUIT'], STARFRUIT_lb, STARFRUIT_ub)


        basket_orders = self.calc_basket(state.order_depths)

        orders['GIFT_BASKET'] += basket_orders['GIFT_BASKET']
        orders['CHOCOLATE'] += basket_orders['CHOCOLATE']
        orders['STRAWBERRIES'] += basket_orders['STRAWBERRIES']
        orders['ROSES'] += basket_orders['ROSES']

        orchid_obs = state.observations.conversionObservations["ORCHIDS"]
        orchid_orders, orchid_conversions = self.calc_orchids(orchid_obs, state.order_depths['ORCHIDS'])
        orders['ORCHIDS'] += orchid_orders
        conversions = self.prev_orchid_conversions
        self.prev_orchid_conversions = orchid_conversions


        coconut_orders = self.calc_coupon(state.order_depths)
        orders['COCONUT'] += coconut_orders['COCONUT']
        orders['COCONUT_COUPON'] += coconut_orders['COCONUT_COUPON']


        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
