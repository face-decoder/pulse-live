import matplotlib.pyplot as plt


class ApexPhaseVisualizer:

    def plot_phases(self,
                    signal: list,
                    apex_indices: list,
                    phases: dict,
                    title: str = "Apex Phases Visualization") -> None:
        """
        Memvisualisasikan sinyal dengan titik apex dan fase apex yang terdeteksi.

        Args:
            signal (list): Sinyal input yang akan divisualisasikan.
            apex_indices (list): Daftar indeks apex yang sudah ditemukan.
            phases (dict): Kamus yang berisi informasi fase apex.
            title (str): Judul grafik.
        """

        _, axes = plt.subplots(figsize=(12, 6))

        # Plot sinyal dan apex points yang terdeteksi
        axes.plot(signal, label='Signal', color='blue')

        # Plot fase yang terdeteksi dengan area berwarna orange
        for apex_index in apex_indices:
            axes.plot(apex_index, signal[apex_index], 'ro', label='Apex' if apex_index == apex_indices[0] else "")

            phase = phases[apex_index]
            axes.axvspan(phase['start'], phase['end'], color='orange', alpha=0.3, label='Apex Phase' if apex_index == apex_indices[0] else "")

        axes.set_title(title)
        axes.set_xlabel('Frame Index')
        axes.set_ylabel('Signal Value')
        axes.legend()

        plt.show()


    def plot_phases_with_actual(self,
                                signal: list,
                                apex_indices: list,
                                phases: dict,
                                actual_phases: dict,
                                title: str = "Apex Phases vs Actual Phases") -> None:
        """
        Memvisualisasikan sinyal dengan titik apex, fase apex yang terdeteksi, dan fase aktual.

        Args:
            signal (list): Sinyal input yang akan divisualisasikan.
            apex_indices (list): Daftar indeks apex yang sudah ditemukan.
            phases (dict): Kamus yang berisi informasi fase apex.
            actual_phases (dict): Kamus yang berisi informasi fase aktual.
            title (str): Judul grafik.
        """

        _, axes = plt.subplots(figsize=(12, 6))

        # Plot sinyal dan apex points yang terdeteksi
        axes.plot(signal, label='Signal', color='blue')
        axes.scatter(apex_indices, [signal[i] for i in apex_indices], color='red', label='Detected Apex Points')

        # Plot fase yang terdeteksi dengan area berwarna hijau
        for apex_index, phase in phases.items():
            axes.axvspan(phase['start'], phase['end'], color='green', alpha=0.3, 
                    label='Detected Phase' if apex_index == apex_indices[0] else "")

        # Plot fase yang sebenarnya dengan garis vertikal
        axes.axvline(actual_phases['onset'], color='orange', linestyle='--', linewidth=2, label='Actual Onset')
        axes.axvline(actual_phases['apex'], color='red', linestyle='--', linewidth=2, label='Actual Apex')
        axes.axvline(actual_phases['offset'], color='purple', linestyle='--', linewidth=2, label='Actual Offset')

        # Tambahkan area berwarna kuning untuk fase yang sebenarnya
        axes.axvspan(actual_phases['onset'], actual_phases['offset'], color='yellow', alpha=0.2, label='Actual Phase')

        axes.legend()
        axes.set_title(title)
        axes.set_xlabel("Frame Index")
        axes.set_ylabel("Optical Flow Magnitude")
        axes.grid(True, alpha=0.3)

        plt.show()