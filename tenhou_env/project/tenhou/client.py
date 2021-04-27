import datetime
import random
import socket
from threading import Thread
from time import sleep
from urllib.parse import quote

# from game.ai.configs.phoenix import PhoenixConfig
from game.client import Client
from mahjong.tile import TilesConverter
from tenhou.decoder import TenhouDecoder
from utils.decisions_logger import MeldPrint
from utils.settings_handler import settings
from utils.statistics import Statistics


class TenhouClient(Client):
    statistics = None
    socket = None
    game_is_continue = True
    looking_for_game = True
    keep_alive_thread = None
    reconnected_messages = None

    decoder = TenhouDecoder()

    _count_of_empty_messages = 0
    _rating_string = None

    def __init__(self, logger, bot_config=None):
        super().__init__(bot_config)
        self.logger = logger
        self.statistics = Statistics()

        self.player.init_logger(logger)

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(1)
        self.socket.connect((settings.TENHOU_HOST, settings.TENHOU_PORT))

    def authenticate(self):
        self._send_message('<HELO name="{}" tid="f0" sx="M" />'.format(quote(settings.USER_ID)))
        messages = self._get_multiple_messages()
        auth_message = messages[0]

        if not auth_message:
            self.logger.info("Auth message wasn't received")
            return False

        # we reconnected to the game
        if "<GO" in auth_message:
            self._send_keep_alive_ping()

            self.logger.info("Successfully reconnected")
            self.reconnected_messages = messages

            selected_game_type = self.decoder.parse_go_tag(auth_message)
            self._set_game_rules(selected_game_type)

            values = self.decoder.parse_names_and_ranks(messages[1])
            self.table.set_players_names_and_ranks(values)

            return True

        rating_string, new_rank_message = self.decoder.parse_hello_string(auth_message)
        self._rating_string = rating_string

        if new_rank_message:
            self.logger.info("Achieved a new rank! \n {}".format(new_rank_message))

        self._send_message(self._pxr_tag())

        # sometimes tenhou send an empty tag after authentication (in tournament mode)
        # and bot thinks that he was not auth
        # to prevent it lets wait a little bit
        # and lets read a group of tags
        continue_reading = True
        counter = 0
        authenticated = False
        while continue_reading:
            messages = self._get_multiple_messages()
            for message in messages:
                if "<LN" in message:
                    authenticated = True
                    continue_reading = False

            counter += 1
            # to avoid infinity loop
            if counter > 10:
                continue_reading = False

        if authenticated:
            self._send_keep_alive_ping()
            self.logger.info("Successfully authenticated")
            return True
        else:
            self.logger.info("Failed to authenticate")
            return False

    def start_game(self):
        log_link = ""

        # play in private or tournament lobby
        if settings.LOBBY != "0":
            if settings.IS_TOURNAMENT:
                self.logger.info("Go to the tournament lobby: {}".format(settings.LOBBY))
                self._send_message('<CS lobby="{}" />'.format(settings.LOBBY))
                self._random_sleep(1, 2)
                self._send_message("<DATE />")
            else:
                self.logger.info("Go to the lobby: {}".format(settings.LOBBY))
                self._send_message('<CHAT text="{}" />'.format(quote("/lobby {}".format(settings.LOBBY))))
                self._random_sleep(1, 2)

        if self.reconnected_messages:
            # we already in the game
            self.looking_for_game = False
            self._send_message("<GOK />")
            self._random_sleep(1, 2)
        else:
            selected_game_type = self._build_game_type()
            game_type = "{},{}".format(settings.LOBBY, selected_game_type)

            if not settings.IS_TOURNAMENT:
                self._random_sleep(2, 5)
                self._send_message('<JOIN t="{}" />'.format(game_type))
                self.logger.info("Looking for the game...")

            start_time = datetime.datetime.now()

            while self.looking_for_game:
                self._random_sleep(1, 2)

                messages = self._get_multiple_messages()
                for message in messages:
                    if "<REJOIN" in message:
                        # game wasn't found, continue to wait
                        self._send_message('<JOIN t="{},r" />'.format(game_type))

                    if "<GO" in message:
                        self._random_sleep(1, 2)
                        self._send_message("<GOK />")
                        self._send_message("<NEXTREADY />")

                        # we had to have it there
                        # because for tournaments we don't know
                        # what exactly game type was set
                        selected_game_type = self.decoder.parse_go_tag(message)
                        process_rules = self._set_game_rules(selected_game_type)
                        if not process_rules:
                            self.logger.error("Hirosima (3 man) is not supported at the moment")
                            self.end_game(success=False)
                            return

                    if "<TAIKYOKU" in message:
                        self.looking_for_game = False
                        game_id, seat = self.decoder.parse_log_link(message)
                        log_link = "http://tenhou.net/0/?log={}&tw={}".format(game_id, seat)

                        self.statistics.game_id = game_id

                        if settings.STAT_SERVER_URL:
                            try:
                                result = self.statistics.send_start_game()
                                self.logger.info(f"Send start game event: {result}")
                            except Exception as e:
                                self.logger.error("Send start game event error", exc_info=e)

                    if "<UN" in message:
                        values = self.decoder.parse_names_and_ranks(message)
                        self.table.set_players_names_and_ranks(values)

                        self.statistics.username = values[0]["name"]

                    if "<LN" in message:
                        self._send_message(self._pxr_tag())

                    # send end game message to api
                    # if settings.TOURNAMENT_API_TOKEN and settings.TOURNAMENT_API_URL:
                    #     if "<CHAT" in message:
                    #         # uname means that other player sent this message, we don't need to parse it
                    #         if not self.decoder.get_attribute_content(message, "uname"):
                    #             text = self.decoder.get_attribute_content(message, "text")
                    #             if text and text.startswith("#END"):
                    #                 try:
                    #                     requests.post(
                    #                         settings.TOURNAMENT_API_URL,
                    #                         data={
                    #                             "api_token": settings.TOURNAMENT_API_TOKEN,
                    #                             "message": text,
                    #                         },
                    #                     )
                    #                     self.logger.error("Successfully sent end game message")
                    #                 except Exception:
                    #                     self.logger.error("Can't sent end game message")
                                        
                current_time = datetime.datetime.now()
                time_difference = current_time - start_time

                if time_difference.seconds > 60 * settings.WAITING_GAME_TIMEOUT_MINUTES:
                    break

        # we wasn't able to find the game in specified time range
        # sometimes it happens and we need to end process
        # and try again later
        if self.looking_for_game:
            self.logger.info("Game is not started. Can't find the game")
            self.end_game()
            return

        self.logger.info("Game started")
        self.logger.info("Log: {}".format(log_link))
        self.logger.info("Players: {}".format(self.table.players))

        main_player = self.table.player

        meld_tile = None
        tile_to_discard = None

        tsumo_win_suggestions = ['t="16"', 't="48"']
        ron_win_suggestions = ['t="8"', 't="9"', 't="10"', 't="11"', 't="12"', 't="13"', 't="15"']
        while self.game_is_continue:
            self._random_sleep(1, 2)

            messages = self._get_multiple_messages()

            if self.reconnected_messages:
                messages = self.reconnected_messages + messages
                self.reconnected_messages = None

            if not messages:
                self._count_of_empty_messages += 1
            else:
                # we had set to zero counter
                self._count_of_empty_messages = 0

            for message in messages:
                if "<INIT" in message or "<REINIT" in message:
                    values = self.decoder.parse_initial_values(message)

                    self.table.init_round(
                        values["round_wind_number"],
                        values["count_of_honba_sticks"],
                        values["count_of_riichi_sticks"],
                        values["dora_indicator"],
                        values["dealer"],
                        values["scores"],
                    )

                    self.logger.info("Round Log: {}&ts={}".format(log_link, self.table.round_number))

                    tiles = self.decoder.parse_initial_hand(message)
                    self.table.player.init_hand(tiles)

                    self.logger.info(self.table)
                    self.logger.info("Players: {}".format(self.table.get_players_sorted_by_scores()))
                    self.logger.info("Dealer: {}".format(self.table.get_player(values["dealer"])))

                if "<REINIT" in message:
                    players = self.decoder.parse_table_state_after_reconnection(message)
                    for x in range(0, 4):
                        player = players[x]
                        for item in player["discards"]:
                            self.table.add_discarded_tile(x, item, False)

                        for item in player["melds"]:
                            if x == 0:
                                tiles = item.tiles
                                main_player.tiles.extend(tiles)
                            self.table.add_called_meld(x, item)

                # draw tile message
                if "<T" in message:
                    self._random_sleep(0.5, 1)

                    drawn_tile = self.decoder.parse_tile(message)
                    self.table.count_of_remaining_tiles -= 1

                    self.logger.info(
                        "Drawn tile: {}".format(
                            TilesConverter.to_one_line_string([drawn_tile], print_aka_dora=self.table.has_aka_dora)
                        )
                    )

                    # we won by self draw (tsumo)
                    if any(i in message for i in tsumo_win_suggestions) and self.player.should_call_win(
                        drawn_tile, is_tsumo=True
                    ):
                        self._random_sleep(0.8, 1.2)
                        self._send_message('<N type="7" />')
                        continue

                    self.player.draw_tile(drawn_tile)

                    if 't="64"' in message and self.player.should_call_kyuushu_kyuuhai():
                        self._random_sleep(1.5, 2)
                        self._send_message('<N type="9" />')
                        continue

                    kan_type = self.player.should_call_kan(drawn_tile, False, main_player.in_riichi)
                    if kan_type:
                        self._random_sleep(1, 1.5)

                        if kan_type == MeldPrint.SHOUMINKAN:
                            meld_type = 5
                            self.logger.info("We upgraded pon to kan!")
                        else:
                            meld_type = 4
                            self.logger.info("We called a closed kan set!")

                        self._send_message('<N type="{}" hai="{}" />'.format(meld_type, drawn_tile))

                        continue

                    if main_player.in_riichi:
                        discarded_tile = drawn_tile
                        # we had to add it to discards, to calculate remaining tiles correctly
                        self.player.discard_tile(drawn_tile, force_tsumogiri=True)
                        self._send_message('<D p="{}"/>'.format(discarded_tile))
                        self.logger.info(
                            "Discard: {}".format(
                                TilesConverter.to_one_line_string(
                                    [discarded_tile], print_aka_dora=self.table.has_aka_dora
                                )
                            )
                        )
                        continue

                    discarded_tile, with_riichi = self.player.discard_tile()

                    # let's call riichi
                    if with_riichi:
                        self._random_sleep(1, 1.5)
                        self._send_message('<REACH hai="{}" />'.format(discarded_tile))
                        main_player.in_riichi = True

                    self._send_message('<D p="{}"/>'.format(discarded_tile))
                    self.logger.info(
                        "Discard: {}".format(
                            TilesConverter.to_one_line_string([discarded_tile], print_aka_dora=self.table.has_aka_dora)
                        )
                    )

                # new dora indicator after kan
                if "<DORA" in message:
                    tile = self.decoder.parse_dora_indicator(message)
                    self.table.add_dora_indicator(tile)
                    self.logger.info(
                        "New dora indicator: {}".format(
                            TilesConverter.to_one_line_string([tile], print_aka_dora=self.table.has_aka_dora)
                        )
                    )

                if "<REACH" in message and 'step="1"' in message:
                    who_called_riichi = self.decoder.parse_who_called_riichi(message)
                    self.table.add_called_riichi_step_one(who_called_riichi)
                    self.logger.info("Riichi called by {} player, step=1".format(who_called_riichi))

                if "<REACH" in message and 'step="2"' in message:
                    who_called_riichi = self.decoder.parse_who_called_riichi(message)
                    self.table.add_called_riichi_step_two(who_called_riichi)
                    self.logger.info("Riichi called by {} player, step=2".format(who_called_riichi))

                # the end of round
                if "<AGARI" in message or "<RYUUKYOKU" in message:
                    gains = self.decoder.parse_gain(message)
                    self.table.gains = gains
                    # self.table.player.ai.collect_experience()
                    self._random_sleep(3, 5)
                    self._send_message("<NEXTREADY />")

                # set was called
                if self.decoder.is_opened_set_message(message):
                    meld = self.decoder.parse_meld(message)
                    self.table.add_called_meld(meld.who, meld)
                    self.logger.info("Meld: {} by {}".format(meld, meld.who))

                    # tenhou confirmed that we called a meld
                    # we had to do discard after this
                    if meld.who == 0:
                        if meld.type != MeldPrint.KAN and meld.type != MeldPrint.SHOUMINKAN:
                            discarded_tile, with_riichi = self.player.discard_tile(tile_to_discard)
                            assert not with_riichi

                            self.player.tiles.append(meld_tile)
                            self._send_message('<D p="{}"/>'.format(discarded_tile))

                # we win by other player's discard
                if any(i in message for i in ron_win_suggestions):
                    is_chankan = False
                    # enemy called shouminkan and we can win there
                    if self.decoder.is_opened_set_message(message):
                        meld = self.decoder.parse_meld(message)
                        tile = meld.called_tile
                        enemy_seat = meld.who
                        is_chankan = True
                    else:
                        tile = self.decoder.parse_tile(message)
                        enemy_seat = self.decoder.get_enemy_seat(message)

                    self._random_sleep(1, 1.5)

                    if main_player.should_call_win(tile, False, enemy_seat, is_chankan):
                        self._send_message('<N type="6" />')
                    else:
                        self._send_message("<N />")

                if self.decoder.is_discarded_tile_message(message):
                    tile = self.decoder.parse_tile(message)

                    # <e21/> - is tsumogiri
                    # <E21/> - discard from the hand
                    if_tsumogiri = message[1].islower()
                    player_seat = self.decoder.get_enemy_seat(message)

                    # open hand suggestions
                    if "t=" in message:
                        # Possible t="" suggestions
                        # 1 pon
                        # 2 kan (it is a closed kan and can be send only to the self draw)
                        # 3 pon + kan
                        # 4 chi
                        # 5 pon + chi
                        # 7 pon + kan + chi

                        # should we call a kan?
                        if 't="3"' in message or 't="7"' in message:
                            if self.player.should_call_kan(tile, True):
                                self._random_sleep(1, 2)

                                # 2 is open kan
                                self._send_message('<N type="2" />')
                                self.logger.info("We called an open kan set!")
                                continue

                        # player with "g" discard is always our kamicha
                        is_kamicha_discard = False
                        if message[1].lower() == "g":
                            is_kamicha_discard = True

                        meld, tile_to_discard = self.player.try_to_call_meld(tile, is_kamicha_discard,
                                                                             int(message.split()[1][3]))
                        if meld:
                            self._random_sleep(1, 2)

                            meld_tile = tile

                            # 1 is pon
                            meld_type = "1"
                            if meld.type == MeldPrint.CHI:
                                # yeah it is 3, not 4
                                # because of tenhou protocol
                                meld_type = "3"

                            tiles = meld.tiles
                            tiles.remove(meld_tile)

                            # try to call a meld
                            self._send_message(
                                '<N type="{}" hai0="{}" hai1="{}" />'.format(meld_type, tiles[0], tiles[1])
                            )
                        # this meld will not improve our hand
                        else:
                            self._send_message("<N />")

                    self.table.add_discarded_tile(player_seat, tile, if_tsumogiri)

                if "owari" in message:
                    values = self.decoder.parse_final_scores_and_uma(message)
                    self.table.set_players_scores(values["scores"], values["uma"])

                if "<PROF" in message:
                    self.game_is_continue = False

            # socket was closed by tenhou
            if self._count_of_empty_messages >= 10:
                self.logger.error("We are getting empty messages from socket. Probably socket connection was closed")
                self.end_game(False)
                return

        self.logger.info("Log: {}".format(log_link))
        self.logger.info("Final results: {}".format(self.table.get_players_sorted_by_scores()))

        
        # we need to finish the game, and only after this try to send statistics
        # if order will be different, tenhou will return 404 on log download endpoint
        self.end_game()

        # sometimes log is not available just after the game
        # let's wait one minute before the statistics update
        if settings.STAT_SERVER_URL:
            sleep(60)
            try:
                result = self.statistics.send_end_game()
                self.logger.info("Statistics sent: {}".format(result))
            except Exception as e:
                self.logger.error("Send end game event error", exc_info=e)

    def end_game(self, success=True):
        self.game_is_continue = False
        if success:
            self._send_message("<BYE />")
            

        if self.keep_alive_thread:
            self.keep_alive_thread.join()

        try:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
        except OSError:
            pass

        if success:
            self.logger.info("End of the game")
        else:
            self.logger.error(f"Game was ended without success, log_id={self.statistics.game_id}")

    def _send_message(self, message):
        # tenhou requires an empty byte in the end of each sending message
        self.logger.debug("Send: {}".format(message))
        message += "\0"
        self.socket.sendall(message.encode())

    def _read_message(self):
        try:
            message = self.socket.recv(4096)
            self.logger.debug("Get: {}".format(message.decode("utf-8").replace("\x00", " ")))
            return message.decode("utf-8")
        except Exception:
            return ""

    def _get_multiple_messages(self):
        # tenhou can send multiple messages in one request
        messages = self._read_message()
        if not messages:
            return []
        messages = messages.split("\x00")
        # last message always is empty after split, so let's exclude it
        messages = messages[0:-1]
        return messages

    def _send_keep_alive_ping(self):
        def send_request():
            while self.game_is_continue:
                self._send_message("<Z />")

                # we can't use sleep(15), because we want to be able
                # end thread in the middle of running
                seconds_to_sleep = 15
                for _ in range(0, seconds_to_sleep * 2):
                    if self.game_is_continue:
                        sleep(0.5)

        self.keep_alive_thread = Thread(target=send_request)
        self.keep_alive_thread.start()

    def _pxr_tag(self):
        # I have no idea why we need to send it, but better to do it
        if settings.IS_TOURNAMENT:
            return '<PXR V="-1" />'

        if settings.USER_ID == "NoName":
            return '<PXR V="1" />'
        else:
            return '<PXR V="9" />'

    def _build_game_type(self):
        # usual case, we specified game type to play
        if settings.GAME_TYPE is not None:
            return settings.GAME_TYPE

        # kyu lobby, hanchan ari-ari
        default_game_type = "9"

        if settings.LOBBY != "0":
            self.logger.error("We can't use dynamic game type and custom lobby. Default game type was set")
            return default_game_type

        if not self._rating_string:
            self.logger.error("For NoName dynamic game type is not available. Default game type was set")
            return default_game_type

        temp = self._rating_string.split(",")
        dan = int(temp[0])
        rate = float(temp[2])
        self.logger.info("Player has {} rank and {} rate".format(TenhouDecoder.RANKS[dan], rate))

        game_type = default_game_type

        # dan lobby, we can play here from 1 kyu
        if dan >= 9:
            game_type = "137"

        # upperdan lobby, we can play here from 4 dan and with 1800+ rate
        if dan >= 13 and rate >= 1800:
            game_type = "41"

        return game_type

    def _set_game_rules(self, game_type):
        """
        Set game related settings and
        return false, if we are trying to play 3 man game
        """
        # need to find a better way to do it
        rules = bin(int(game_type)).replace("0b", "")
        while len(rules) != 8:
            rules = "0" + rules

        is_hanchan = rules[4] == "1"
        is_open_tanyao = rules[5] == "0"
        is_aka = rules[6] == "0"
        is_hirosima = rules[3] == "1"

        if is_hirosima:
            return False

        self.table.has_aka_dora = is_aka
        self.table.has_open_tanyao = is_open_tanyao

        self.logger.info("Game settings:")
        self.logger.info("Aka dora: {}".format(self.table.has_aka_dora))
        self.logger.info("Open tanyao: {}".format(self.table.has_open_tanyao))
        self.logger.info("Game type: {}".format(is_hanchan and "hanchan" or "tonpusen"))

        return True

    def _random_sleep(self, min_sleep, max_sleep):
        sleep(random.uniform(min_sleep, max_sleep))
