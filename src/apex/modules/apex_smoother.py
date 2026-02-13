class ApexSmoother:

    WINDOW_LENGTH_PERCENTAGE = 0.1

    @staticmethod
    def calculate_window_length(length: int) -> int:
        """
        Menghitung panjang window yang sesuai untuk melakukan perataan pada magnitudo sinyal.

        Args:
            length (int): Panjang sinyal input.

        Returns:
            int: Panjang window yang digunakan untuk perataan.
        """

        window_length = int(length / ApexSmoother.WINDOW_LENGTH_PERCENTAGE)

        # Jika panjang window genap, tambahkan 1 agar menjadi ganjil
        # Hal ini untuk savgol filter yang memerlukan panjang window ganjil
        if window_length % 2 == 0: window_length += 1

        # Membatasi panjang window antara 5 hingga 51
        # Strategi ini untuk menghindari over-smoothing pada sinyal pendek
        window_length = max(5, min(window_length, 51))

        return window_length
    

    @staticmethod
    def calculate_polyorder(window_length: int) -> int:
        """
        Menghitung orde polinomial yang sesuai untuk savgol filter berdasarkan panjang window.

        Args:
            window_length (int): Panjang window yang digunakan untuk perataan.

        Returns:
            int: Orde polinomial yang digunakan dalam savgol filter.
        """

        match window_length:
            case wl if wl <= 7:
                return 2
            case wl if wl <= 15:
                return 3
            case _:
                return 4
